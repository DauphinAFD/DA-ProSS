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
        depth_out = classifier['D'](tgt_feats_depth)
        seg_out = classifier['S'](tgt_feats_seg)
        trans_out = classifier['T'](tgt_feats_trans)
        final_out = classifier['FS'](tgt_feats_final)

    seg_out = F.interpolate(seg_out, size=size, mode='bilinear', align_corners=True)
    seg_out = F.softmax(seg_out, dim=1)

    
    depth_out = F.interpolate(depth_out, size=size, mode='bilinear', align_corners=True)
    trans_out = F.interpolate(trans_out, size=size, mode='bilinear', align_corners=True)
    trans_out = F.softmax(trans_out, dim=1)

    final_out = F.interpolate(final_out, size=size, mode='bilinear', align_corners=True)
    final_out = F.softmax(final_out, dim=1)
    # if flip:
    #     output = (output[0] + output[1].flip(2)) / 2
    # else:
    #     output = output[0]
    return seg_out.squeeze(0).argmax(0), depth_out.squeeze(0), trans_out.squeeze(0).argmax(0), final_out.squeeze(0).argmax(0)

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
        tgt_input = Fnorn.resize(tgt_input, (512,1024), Image.BICUBIC)
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
    classifier['D'].to(device)
    classifier['S'].to(device)
    classifier['T'].to(device)
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
    classifier['D'].eval()
    classifier['S'].eval()
    classifier['T'].eval()
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
        output_seg_folder = os.path.join(cfg.OUTPUT_DIR, "seg_output")
        output_depth_folder = os.path.join(cfg.OUTPUT_DIR, "depth_output")
        output_trans_folder = os.path.join(cfg.OUTPUT_DIR, "trans_output")
        output_final_folder = os.path.join(cfg.OUTPUT_DIR, "final_output")
        mkdir(output_seg_folder)
        mkdir(output_depth_folder)
        mkdir(output_trans_folder)
        mkdir(output_final_folder)
        


    # src_data = build_dataset(cfg, mode='pseudo', is_source=True)
    test_data = build_dataset(cfg, mode='pseudo', is_source=False)
    src_train_sampler = None

    tgt_loader = torch.utils.data.DataLoader(test_data, batch_size=cfg.SOLVER.BATCH_SIZE_VAL,
                                              shuffle=False, num_workers=0,
                                              pin_memory=True, sampler=None)
    # src_loader = torch.utils.data.DataLoader(src_data, batch_size=cfg.SOLVER.BATCH_SIZE,
    #                                                shuffle=(src_train_sampler is None), num_workers=0,
    #                                                pin_memory=True, sampler=src_train_sampler, drop_last=True)
    
    k = 0

    for index, (tgt_input, tgt_label ,_, name) in enumerate(tgt_loader):

        if index % 100 == 0:
            logger.info("{} processed".format(index))

        tgt_input = tgt_input.cuda(non_blocking=True)
        tgt_label = tgt_label.cuda(non_blocking=True).long()


        tgt_copy = tgt_input.clone()
        tgt_copy = Fnorn.resize(tgt_copy, (512,1024), Image.BICUBIC)
        tgt_copy = Fnorn.normalize(tgt_copy, mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)

        seg_out, depth_out, trans_out, final_out = inference(feature_extractor, classifier, tgt_copy, tgt_label, flip=False)

        
        # Iteration Controller
        k += 1
        
        min = depth_out.min()
        max = depth_out.max()
        max -= min
        depth_img = torch.round(((depth_out - min)/max) * 255.0)


        mask_filename = name[0] if len(name[0].split("/")) < 2 else name[0].split("/")[-1]

        mask_seg = get_color_pallete(seg_out.cpu().numpy(), "city")
        mask_seg.save(os.path.join(output_seg_folder, mask_filename))

        depth_out = Image.fromarray((depth_img.squeeze(0)).cpu().numpy().astype(np.uint8))
        depth_out.save(os.path.join(output_depth_folder, mask_filename))


        mask_trans = get_color_pallete(trans_out.cpu().numpy(), "city")
        mask_trans.save(os.path.join(output_trans_folder, mask_filename))

        mask_final = get_color_pallete(final_out.cpu().numpy(), "city")
        mask_final.save(os.path.join(output_final_folder, mask_filename))






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