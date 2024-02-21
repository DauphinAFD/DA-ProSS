import argparse
import os
import logging
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from time import sleep
from PIL import Image
import cv2
from core.datasets import transform

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as Fnorn
import torch.backends.cudnn
import kornia
import matplotlib.pyplot as plt
from core.configs import cfg
from core.datasets import build_dataset
from core.models import build_feature_extractor, build_classifier
from core.utils.misc import mkdir, AverageMeter, intersectionAndUnionGPU, get_color_pallete
from core.utils.logger import setup_logger
from core.utils.pseudo_label import PseudoLabel

def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict

def inference(feature_extractor, classifier, image, label, flip=True):
    size = label.shape[-2:]
    if flip:
        image = torch.cat([image, torch.flip(image, [3])], 0)
    with torch.no_grad():
        tgt_feats_back = feature_extractor['B'](image)
        tgt_feats_seg = feature_extractor['S'](tgt_feats_back)
        tgt_feats_depth = feature_extractor['D'](tgt_feats_back)
        tgt_feats_trans = feature_extractor['T'](tgt_feats_depth)
        tgt_feats_final = feature_extractor['FS'](tgt_feats_seg, tgt_feats_trans)
        output = classifier(tgt_feats_final)
    output = F.interpolate(output, size=size, mode='bilinear', align_corners=True)
    output = F.softmax(output, dim=1)
    if flip:
        output = (output[0] + output[1].flip(2)) / 2
    else:
        output = output[0]
    return output.unsqueeze(dim=0)

def transform_color(pred):
    synthia_to_city = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        8: 8,
        9: 10,
        10: 11,
        11: 12,
        12: 13,
        13: 15,
        14: 17,
        15: 18,
    }
    label_copy = 255 * np.ones(pred.shape, dtype=np.float32)
    for k, v in synthia_to_city.items():
        label_copy[pred == k] = v
    return label_copy.copy()

void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33,]

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

ignore_index = 250
class_map = dict(zip(valid_classes, range(19)))

def encode_segmap(mask):
        # Put all void classes to zero
        for _voidc in void_classes:
            mask[mask == _voidc] = ignore_index
        for _validc in valid_classes:
            mask[mask == _validc] = class_map[_validc]
        return mask

def generate_class_mask(pred, classes):
    pred, classes = torch.broadcast_tensors(pred.unsqueeze(0), classes.unsqueeze(1).unsqueeze(2))
    N = pred.eq(classes).sum(0)
    return N

def oneMix(mask, data = None, target = None):
    #Mix
    if not (data is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], data[0])
        data = (stackedMask0*data[0]+(1-stackedMask0)*data[1]).unsqueeze(0)
    if not (target is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], target[0])
        target = (stackedMask0*target[0]+(1-stackedMask0)*target[1]).unsqueeze(0)
    return data, target

def strongTransform(parameters, data=None, target=None):
    assert ((data is not None) or (target is not None))
    data, target = oneMix(mask = parameters["Mix"], data = data, target = target)
    # data, target = colorJitter(colorJitter = parameters["ColorJitter"], img_mean = torch.from_numpy(IMG_MEAN.copy()).cuda(), data = data, target = target)
    # data, target = transformsgpu.gaussian_blur(blur = parameters["GaussianBlur"], data = data, target = target)
    # data, target = transformsgpu.flip(flip = parameters["flip"], data = data, target = target)
    
    return data, target

def get_threshold(cfg):
    logger = logging.getLogger("pseudo_label.trainer")
    logger.info("Start inference on target dataset and get threshold of each class")
    NUM_CLASSES = {"S": 16, "D": 1}
    TASKNAMES = ["S", "D"]
    device = torch.device(cfg.MODEL.DEVICE)
    feature_extractor = build_feature_extractor(cfg, TASKNAMES)
    feature_extractor['B'].to(device)
    feature_extractor['S'].to(device)
    feature_extractor['D'].to(device)
    feature_extractor['T'].to(device)
    feature_extractor['FS'].to(device)

    classifier = build_classifier(cfg, TASKNAMES, NUM_CLASSES)
    classifier['FS'].to(device)

    if cfg.resume:
        logger.info("Loading checkpoint from {}".format(cfg.resume))
        checkpoint = torch.load(cfg.resume, map_location=torch.device('cpu'))
        model_weights = strip_prefix_if_present(checkpoint['feature_extractor'], 'module.')
        feature_extractor.load_state_dict(model_weights)
        classifier_weights = strip_prefix_if_present(checkpoint['classifier'], 'module.')
        classifier.load_state_dict(classifier_weights)

    feature_extractor['B'].eval()
    feature_extractor['S'].eval()
    feature_extractor['D'].eval()
    feature_extractor['T'].eval()
    feature_extractor['FS'].eval()
    classifier['FS'].eval()

    torch.cuda.empty_cache()
    tgt_train_data = build_dataset(cfg, mode='test', is_source=False)
    tgt_train_loader = torch.utils.data.DataLoader(tgt_train_data,
                                                   batch_size=cfg.SOLVER.BATCH_SIZE_VAL,
                                                   shuffle=False,
                                                   num_workers=4,
                                                   pin_memory=True,
                                                   sampler=None,
                                                   drop_last=False)

    cpseudo_label = PseudoLabel(cfg)
    for batch in tqdm(tgt_train_loader):
        x, _,_, name = batch
        tgt_input = x.cuda(non_blocking=True)
        tgt_size = tgt_input.shape[-2:]
        with torch.no_grad():
            tgt_feats_back = feature_extractor['B'](tgt_input)
            tgt_feats_seg = feature_extractor['S'](tgt_feats_back)
            tgt_feats_depth = feature_extractor['D'](tgt_feats_back)
            tgt_feats_trans = feature_extractor['T'](tgt_feats_depth)
            tgt_feats_final = feature_extractor['FS'](tgt_feats_seg, tgt_feats_trans)

            output = classifier['FS'](tgt_feats_final)
        output = F.interpolate(output, size=tgt_size, mode='bilinear', align_corners=True)
        cpseudo_label.update_pseudo_label(output)
    thres_const = cpseudo_label.get_threshold_const(thred=0.9, percent=cfg.MODEL.THRESHOLD_PERCENT)
    cpseudo_label.save_results()

    return thres_const

def test(cfg, thres_const):
    logger = logging.getLogger("pseudo_label.tester")
    logger.info("Start testing")
    NUM_CLASSES = {"S": 16, "D": 1}
    TASKNAMES = ["S", "D"]
    device = torch.device(cfg.MODEL.DEVICE)

    feature_extractor = build_feature_extractor(cfg, TASKNAMES)
    feature_extractor['B'].to(device)
    feature_extractor['S'].to(device)
    feature_extractor['D'].to(device)
    feature_extractor['T'].to(device)
    feature_extractor['FS'].to(device)

    classifier = build_classifier(cfg, TASKNAMES, NUM_CLASSES)
    classifier['FS'].to(device)

    if cfg.resume:
        logger.info("Loading checkpoint from {}".format(cfg.resume))
        checkpoint = torch.load(cfg.resume, map_location=torch.device('cpu'))
        feature_extractor_weights = strip_prefix_if_present(checkpoint['feature_extractor'], 'module.')
        feature_extractor.load_state_dict(feature_extractor_weights)
        classifier_weights = strip_prefix_if_present(checkpoint['classifier'], 'module.')
        classifier.load_state_dict(classifier_weights)

    feature_extractor['B'].eval()
    feature_extractor['S'].eval()
    feature_extractor['D'].eval()
    feature_extractor['T'].eval()
    feature_extractor['FS'].eval()
    classifier['FS'].eval()

    intersection_meter1 = AverageMeter()
    union_meter1 = AverageMeter()
    target_meter1 = AverageMeter()

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    torch.cuda.empty_cache()
    assert cfg.DATASETS.TEST == 'cityscapes_train'
    dataset_name = cfg.DATASETS.TEST
    output_folder = '.'
    if cfg.OUTPUT_DIR:
        output_mask_folder = os.path.join(cfg.OUTPUT_DIR, "mixed_soft_labels")
        output_img_folder = os.path.join(cfg.OUTPUT_DIR, "mixed_images")
        output_folder = os.path.join(cfg.OUTPUT_DIR, "soft_labels", dataset_name)
        mkdir(output_folder)
        mkdir(output_mask_folder)
        mkdir(output_img_folder)


    src_data = build_dataset(cfg, mode='pseudo', is_source=True)
    test_data = build_dataset(cfg, mode='pseudo', is_source=False)
    src_train_sampler = None

    tgt_loader = torch.utils.data.DataLoader(test_data, batch_size=cfg.SOLVER.BATCH_SIZE_VAL,
                                              shuffle=False, num_workers=0,
                                              pin_memory=True, sampler=None)
    src_loader = torch.utils.data.DataLoader(src_data, batch_size=cfg.SOLVER.BATCH_SIZE,
                                                   shuffle=(src_train_sampler is None), num_workers=0,
                                                   pin_memory=True, sampler=src_train_sampler, drop_last=True)
    
    k = 0

    for index, ((src_input, src_label, _ , _ ), (tgt_input, tgt_label ,_, name)) in enumerate(zip(src_loader,tgt_loader)):

        if index % 100 == 0:
            logger.info("{} processed".format(index))

        # src_input, src_label, _, name = batch
        src_input = src_input.cuda(non_blocking=True)
        src_label = src_label.cuda(non_blocking=True).long()

        tgt_input = tgt_input.cuda(non_blocking=True)
        tgt_label = tgt_label.cuda(non_blocking=True).long()

        

        tgt_copy = tgt_input.clone()
        tgt_copy = Fnorn.resize(tgt_copy, (512,1024), Image.BICUBIC)
        tgt_copy = Fnorn.normalize(tgt_copy, mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)

        pred = inference(feature_extractor, classifier['FS'], tgt_copy, tgt_label, flip=False)

        pred = pred.cpu().numpy().squeeze()
        pred_max = np.max(pred, 0)
        pred_label = pred.argmax(0)
        pred_copy = pred_label.copy()
        for i in range(cfg.MODEL.NUM_CLASSES):
            pred_label[(pred_max < thres_const[i]) * (pred_label == i)] = 255

        pred_label = torch.Tensor(pred_label).cuda()
        for j in range(len(src_input)) :

            classes = torch.unique(src_label[j])
            use_list = [5, 7, 10, 11, 12, 14, 15]
            classes = torch.Tensor([item for item in classes if item in use_list]).cuda()
            nclasses = classes.shape[0]

            if nclasses > 0:
                classes = (classes[torch.Tensor(np.random.choice(nclasses, np.random.randint(1 , max(nclasses-2,2) ),replace=False)).long()])
                mask_list = []
                for i in range(len(classes)):
                    mask_list.append(generate_class_mask(src_label[j], classes[i].unsqueeze(0)).unsqueeze(0))


                for i in range(len(mask_list)):

                    masked_img = mask_list[i]
                    masked_img = masked_img.squeeze(0).cpu().numpy()
                    masked_img = masked_img.astype(np.uint8)
                    masked_img = np.multiply(masked_img,255)
                    contours, _ = cv2.findContours(masked_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    areas = []
                    for cnt in contours:
                        x, y, w, h = cv2.boundingRect(cnt)
                        areas.append(w*h)

                    max_area = max(areas)
                    max_area = max(max_area, 2500)

                    for cnt in contours:
                        x, y, w, h = cv2.boundingRect(cnt)
                        if(h*w < max_area/10):
                            cv2.rectangle(masked_img, (x, y), (x+w, y+h), (0, 0, 0), -1)
                    
                    mask_list[i] = torch.Tensor(masked_img/255)

                final_mask = torch.zeros(mask_list[0].shape)

                for i in range(len(mask_list)):
                    final_mask = torch.add(final_mask, mask_list[i])
            else :
                final_mask = torch.ones((src_label[j].shape))

            final_mask = final_mask.cuda()
            strong_parameters = {"Mix": final_mask.unsqueeze(0)}
            mixed_img, _ = strongTransform(strong_parameters, data = torch.cat((src_input[j].unsqueeze(0), tgt_input)))
            mixed_label, _ = strongTransform(strong_parameters, data = torch.cat((src_label[j].unsqueeze(0), pred_label.unsqueeze(0))))

            # Iteration Controller
            k += 1
            
            # save the pseudo label
            mask = get_color_pallete(mixed_label.squeeze(0).cpu().numpy(), "city")
            mask_filename = 'label_' + str(k).zfill(5) + '.png'
            mask.save(os.path.join(output_mask_folder, mask_filename))

            mixed_img = Image.fromarray((mixed_img * 255).squeeze(0).permute(1,2,0).cpu().numpy().astype('uint8'))
            img_filename = 'image_' + str(k).zfill(5) + '.png'
            mixed_img.save(os.path.join(output_img_folder, img_filename))

        mask = get_color_pallete(pred_label.cpu().numpy(), "city")
        mask_filename = name[0] if len(name[0].split("/")) < 2 else name[0].split("/")[-1]
        mask.save(os.path.join(output_folder, mask_filename))

        pred_copy = torch.Tensor(pred_copy).unsqueeze(dim = 0).cuda()
        intersection, union, target = intersectionAndUnionGPU(pred_copy, tgt_label, cfg.MODEL.NUM_CLASSES, cfg.INPUT.IGNORE_LABEL)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    logger.info('Val result label: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    for i in range(cfg.MODEL.NUM_CLASSES):
        logger.info(
            '{} {} iou/accuracy: {:.4f}/{:.4f}.'.format(i, test_data.trainid2name[i], iou_class[i], accuracy_class[i]))



def main():
    parser = argparse.ArgumentParser(description="PyTorch Target Pseudo Label Testing")
    parser.add_argument("-cfg",
                        "--config-file",
                        default="",
                        metavar="FILE",
                        help="path to config file",
                        type=str,
                        )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = cfg.OUTPUT_DIR
    if save_dir:
        mkdir(save_dir)
    logger = setup_logger("pseudo_label", save_dir, 0)
    logger.info(cfg)

    logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))

    # thres_const = get_threshold(cfg)
    thres_const = np.load('results/synthia/r101_g2c_ours_proca/thres_const.npy')
    test(cfg, thres_const)

if __name__ == "__main__":
    main()