import argparse
import os
import datetime
import logging
import time
import numpy as np
import wandb
import torch
import kornia
import torch.nn as nn
import random
import torch.nn.functional as F
import torch.utils.data
import matplotlib.pyplot as plt

from core.configs import cfg
from core.datasets import build_dataset
from core.models import build_feature_extractor, build_classifier
from core.solver import adjust_learning_rate
from core.utils.misc import mkdir, AverageMeter, intersectionAndUnionGPU
from core.utils.logger import setup_logger
from core.utils.metric_logger import MetricLogger
from core.utils.loss import BerhuLoss,CrossEntropyLoss2dPixelWiseWeighted
import warnings
warnings.filterwarnings('ignore')

def strip_prefix_if_present(state_dict, prefix):
    from collections import OrderedDict
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith(prefix + 'layer5'):
            continue
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict


void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33,]

IMG_MEAN = np.array((122.67891434, 116.66876762, 104.00698793), dtype=np.float32)

ignore_index = 250
class_map = dict(zip(valid_classes, range(19)))

def encode_segmap(mask):
        # Put all void classes to zero
        for _voidc in void_classes:
            mask[mask == _voidc] = ignore_index
        for _validc in valid_classes:
            mask[mask == _validc] = class_map[_validc]
        return mask

def colorJitter(colorJitter, img_mean, data = None, target = None, s=0.5):
    # s is the strength of colorjitter
    #colorJitter
    if not (data is None):
        if data.shape[1]==3:
            if colorJitter > 0.2:
                # print("inside Color jitter")
                img_mean, _ = torch.broadcast_tensors(img_mean.unsqueeze(0).unsqueeze(2).unsqueeze(3), data)
                seq = nn.Sequential(kornia.augmentation.ColorJitter(brightness=s,contrast=s,saturation=s,hue=0.2))
                data = (data+img_mean)/255
                data = seq(data)
                data = (data*255-img_mean).float()
    return data, target

def gaussian_blur(blur, data = None, target = None):
    if not (data is None):
        if data.shape[1]==3:
            if blur > 0.5:
                sigma = np.random.uniform(0.15,1.15)
                kernel_size_y = int(np.floor(np.ceil(0.1 * data.shape[2]) - 0.5 + np.ceil(0.1 * data.shape[2]) % 2))
                kernel_size_x = int(np.floor(np.ceil(0.1 * data.shape[3]) - 0.5 + np.ceil(0.1 * data.shape[3]) % 2))
                kernel_size = (kernel_size_y, kernel_size_x)
                seq = nn.Sequential(kornia.filters.GaussianBlur2d(kernel_size=kernel_size, sigma=(sigma, sigma)))
                data = seq(data)
    return data, target

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
    # print(parameters)
    data, target = oneMix(mask = parameters["Mix"], data = data, target = target)
    data, target = colorJitter(colorJitter = parameters["ColorJitter"], img_mean = torch.from_numpy(IMG_MEAN.copy()).cuda(), data = data, target = target)
    data, target = gaussian_blur(blur = parameters["GaussianBlur"], data = data, target = target)
    # data, target = transformsgpu.flip(flip = parameters["flip"], data = data, target = target)
    
    return data, target

def create_ema_model(feature_extractor,classifier,TASKNAMES, NUM_CLASSES):

    ema_feature_extractor = build_feature_extractor(cfg, TASKNAMES)
    ema_classifier = build_classifier(cfg, TASKNAMES, NUM_CLASSES)

    # For Feature Extractor
    for param in ema_feature_extractor.parameters():
        param.detach_()
    feature_mp = list(feature_extractor.parameters())
    feature_mcp = list(ema_feature_extractor.parameters())
    n = len(feature_mp)
    for i in range(0, n):
        feature_mcp[i].data[:] = feature_mp[i].data[:].clone()

    # For Classifier
    for param in ema_classifier.parameters():
        param.detach_()
    classifier_mp = list(classifier.parameters())
    classifier_mcp = list(ema_classifier.parameters())
    n = len(classifier_mp)
    for i in range(0, n):
        classifier_mcp[i].data[:] = classifier_mp[i].data[:].clone()

    return ema_feature_extractor, ema_classifier

def update_ema_variables(ema_feature_extractor, ema_classifier, feature_extractor,classifier, alpha_teacher, iteration):
    # Use the "true" average until the exponential average is more correct
    alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)
    
    for ema_param_feat, param_feat in zip(ema_feature_extractor.parameters(), feature_extractor.parameters()):
        ema_param_feat.data[:] = alpha_teacher * ema_param_feat[:].data[:] + (1 - alpha_teacher) * param_feat[:].data[:]

    for ema_param_classi, param_classi in zip(ema_feature_extractor.parameters(), feature_extractor.parameters()):
        ema_param_classi.data[:] = alpha_teacher * ema_param_classi[:].data[:] + (1 - alpha_teacher) * param_classi[:].data[:]
    
    return ema_feature_extractor, ema_classifier


def train(cfg, local_rank, distributed, logger):
    logger.info("Start training")

    NUM_CLASSES = {"S": 16, "D": 1}
    TASKNAMES = ["S", "D"]

    run = wandb.init(
        project = 'DA-ProSS',
        config = cfg
    )

    feature_extractor = build_feature_extractor(cfg, TASKNAMES)
    classifier = build_classifier(cfg, TASKNAMES, NUM_CLASSES)

    device = torch.device(cfg.MODEL.DEVICE)
    # feature_extractor.to(device)
    feature_extractor['B'].to(device)
    # classifier.to(device)
    classifier['S'].to(device)

    ema_feature_extractor, ema_classifier = create_ema_model(feature_extractor,classifier,TASKNAMES, NUM_CLASSES)
    ema_feature_extractor['B'].to(device)
    ema_classifier['S'].to(device)

    batch_size = cfg.SOLVER.BATCH_SIZE
    
    if local_rank == 0:
        print(feature_extractor)
        print(classifier)

    param_groups_fea = [{'params': feature_extractor['B'].parameters(), 'lr': cfg.SOLVER.BASE_LR}]
                        # {'params': feature_extractor['D'].parameters(), 'lr': cfg.SOLVER.BASE_LR},
                        # {'params': feature_extractor['T'].parameters(), 'lr': 10*cfg.SOLVER.BASE_LR},
                        # {'params': feature_extractor['S'].parameters(), 'lr': 10*cfg.SOLVER.BASE_LR},
                        # {'params': feature_extractor['FS'].parameters(), 'lr': 10*cfg.SOLVER.BASE_LR}]
    
    optimizer_fea = torch.optim.SGD(param_groups_fea, momentum=cfg.SOLVER.MOMENTUM,
                                    weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer_fea.zero_grad()

    param_groups_cls = [{'params': classifier['S'].parameters(), 'lr': 1*cfg.SOLVER.BASE_LR}]
                    # {'params': classifier['T'].parameters(), 'lr': 10*cfg.SOLVER.BASE_LR},
                    # {'params': classifier['S'].parameters(), 'lr': 10*cfg.SOLVER.BASE_LR},
                    # {'params': classifier['FS'].parameters(), 'lr': 10*cfg.SOLVER.BASE_LR}]
    
    optimizer_cls = torch.optim.SGD(param_groups_cls, momentum=cfg.SOLVER.MOMENTUM,
                                    weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    
    # optimizer_cls = torch.optim.SGD(classifier['S'].parameters(), lr = 10*cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM,
    #                                 weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer_cls.zero_grad()

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = local_rank == 0

    iteration = 0

    if cfg.resume:
        logger.info("Loading checkpoint from {}".format(cfg.resume))
        checkpoint = torch.load(cfg.resume, map_location=torch.device('cpu'))
        model_weights = checkpoint['feature_extractor'] if distributed else strip_prefix_if_present(
            checkpoint['feature_extractor'], 'module.')
        feature_extractor.load_state_dict(model_weights)
        classifier_weights = checkpoint['classifier'] if distributed else strip_prefix_if_present(
            checkpoint['classifier'], 'module.')
        classifier.load_state_dict(classifier_weights)

        ema_model_weights = checkpoint['ema_feature_extractor'] if distributed else strip_prefix_if_present(
            checkpoint['ema_feature_extractor'], 'module.')
        ema_feature_extractor.load_state_dict(ema_model_weights)
        ema_classifier_weights = checkpoint['ema_classifier'] if distributed else strip_prefix_if_present(
            checkpoint['ema_classifier'], 'module.')
        ema_classifier.load_state_dict(ema_classifier_weights)
        if "optimizer_fea" in checkpoint:
            logger.info("Loading optimizer_fea from {}".format(cfg.resume))
            optimizer_fea.load_state_dict(checkpoint['optimizer_fea'])
        if "optimizer_cls" in checkpoint:
            logger.info("Loading optimizer_cls from {}".format(cfg.resume))
            optimizer_cls.load_state_dict(checkpoint['optimizer_cls'])
        if "iteration" in checkpoint:
            iteration = checkpoint['iteration']

    src_train_data = build_dataset(cfg, mode='train', is_source=True)
    tgt_train_data = build_dataset(cfg, mode='train', is_source=False)
    
    src_train_sampler = None
    tgt_train_sampler = None

    src_train_loader = torch.utils.data.DataLoader(src_train_data, batch_size=batch_size,
                                                   shuffle=(src_train_sampler is None), num_workers=0,
                                                   pin_memory=True, sampler=src_train_sampler, drop_last=True)
    tgt_train_loader = torch.utils.data.DataLoader(tgt_train_data, batch_size=batch_size,
                                                   shuffle=(tgt_train_sampler is None), num_workers=0,
                                                   pin_memory=True, sampler=tgt_train_sampler, drop_last=True)
    # tgt_train_loader = iter(tgt_train_loader)


    ce_criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    berhu_criterion = BerhuLoss().cuda()
    unlabeled_loss = CrossEntropyLoss2dPixelWiseWeighted(ignore_index=255).cuda()


    max_iters = cfg.SOLVER.MAX_ITER
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    
    feature_extractor['B'].train()
    # feature_extractor['D'].train()
    # feature_extractor['S'].train()
    # feature_extractor['T'].train()
    # feature_extractor['FS'].train()
    
    classifier['S'].train()
    # classifier['D'].train()
    # classifier['T'].train()
    # classifier['FS'].train()

    ema_feature_extractor['B'].train()
    ema_classifier['S'].train()

    start_training_time = time.time()
    end = time.time()

    best_mIoU = 0
    best_iteration = 0

    for i in range(cfg.SOLVER.STOP_ITER):
    # for i, (src_input, src_label, _, _) in enumerate(train_loader):
        data_time = time.time() - end

        optimizer_fea.zero_grad()
        optimizer_cls.zero_grad()
        current_lr = adjust_learning_rate(cfg.SOLVER.LR_METHOD, cfg.SOLVER.BASE_LR, iteration, max_iters,
                                          power=cfg.SOLVER.LR_POWER)
        optimizer_fea.param_groups[0]['lr'] = current_lr

        for index in range(1, len(optimizer_fea.param_groups)):
            optimizer_fea.param_groups[index]['lr'] = current_lr * 10
        for index in range(len(optimizer_cls.param_groups)):
            optimizer_cls.param_groups[index]['lr'] = current_lr * 10

        try:
            tgt_batch = next(tgt_train_loader)
            if tgt_batch[0].shape[0] != batch_size:
                tgt_batch = next(tgt_train_loader)
        except:
            tgt_train_loader = iter(tgt_train_loader)
            tgt_batch = next(tgt_train_loader)

        tgt_input, tgt_label, _, name = tgt_batch
        
        tgt_input = tgt_input.cuda(non_blocking=True)
        tgt_label = tgt_label.cuda(non_blocking=True).long()

        size = tgt_label.shape[-2:]
        
        tgt_feats_back = feature_extractor['B'](tgt_input)
        pred_seg_final = classifier['S'](tgt_feats_back, size)

        # pseudo_label = torch.softmax(pred_seg_final.detach(), dim=1)
        # max_probs, pred_label = torch.max(pseudo_label, dim=1)
        # max_probs, pred_label = torch.max(tgt_label, dim=1)


        L_1 = ce_criterion(pred_seg_final, tgt_label)

        # if True : 
        try:
            src_batch = next(src_train_loader)
            if src_batch[0].shape[0] != batch_size:
                src_batch = next(src_train_loader)
        except:
            # epochs_since_start = epochs_since_start + 1
            # print('Epochs since start: ',epochs_since_start)
            src_train_loader = iter(src_train_loader)
            src_batch = next(src_train_loader)

        src_input, src_label, _, name = src_batch

        src_input = src_input.cuda(non_blocking=True)
        src_label = src_label.cuda(non_blocking=True).long()

        for image_i in range(batch_size) :

            classes = torch.unique(src_label[image_i])
            nclasses = classes.shape[0]

            # if nclasses > 0:
            classes = (classes[torch.Tensor(np.random.choice(nclasses, int((nclasses+nclasses%2)/2),replace=False)).long()]).cuda()
            if image_i == 0:
                MixMask0 = generate_class_mask(src_label[image_i], classes).unsqueeze(0).cuda()
            else:
                MixMask1 = generate_class_mask(src_label[image_i], classes).unsqueeze(0).cuda()

        
        
        # # strong_parameters["flip"] = 0
        strong_parameters = {"Mix": MixMask0}
        strong_parameters["ColorJitter"] = random.uniform(0, 1)
        strong_parameters["GaussianBlur"] = random.uniform(0, 1)
        
        mixed_img_0, _ = strongTransform(strong_parameters, data = torch.cat((src_input[0].unsqueeze(0), tgt_input[0].unsqueeze(0))))
        strong_parameters["Mix"] = MixMask1
        mixed_img_1, _ = strongTransform(strong_parameters, data = torch.cat((src_input[1].unsqueeze(0), tgt_input[1].unsqueeze(0))))
        mixed_input = torch.cat((mixed_img_0,mixed_img_1))

    

        mixed_feats_back = ema_feature_extractor['B'](mixed_input)
        mixed_seg_final = ema_classifier['S'](mixed_feats_back, size)


        strong_parameters["Mix"] = MixMask0
        strong_parameters["ColorJitter"] = 0
        strong_parameters["GaussianBlur"] = 0

        mixed_label_0, _ = strongTransform(strong_parameters, data = torch.cat((src_label[0].unsqueeze(0), tgt_label[0].unsqueeze(0))))
        strong_parameters["Mix"] = MixMask1
        mixed_label_1, _ = strongTransform(strong_parameters, data = torch.cat((src_label[1].unsqueeze(0), tgt_label[1].unsqueeze(0))))
        
        mixed_label = torch.cat((mixed_label_0,mixed_label_1)).long()

        # unlabeled_weight = torch.sum(max_probs.ge(0.968).long() == 1).item() / np.size(np.array(mixed_label.cpu()))
        # pixelWiseWeight = unlabeled_weight * torch.ones(max_probs.shape).cuda()

        # onesWeights = torch.ones((pixelWiseWeight.shape)).cuda()
        # strong_parameters["Mix"] = MixMask0
        # _, pixelWiseWeight0 = strongTransform(strong_parameters, target = torch.cat((onesWeights[0].unsqueeze(0),pixelWiseWeight[0].unsqueeze(0))))
        # strong_parameters["Mix"] = MixMask1
        # _, pixelWiseWeight1 = strongTransform(strong_parameters, target = torch.cat((onesWeights[1].unsqueeze(0),pixelWiseWeight[1].unsqueeze(0))))
        # pixelWiseWeight = torch.cat((pixelWiseWeight0,pixelWiseWeight1)).cuda()


        L_u = ce_criterion(mixed_seg_final, mixed_label) 
        # L_u =  0.0 * unlabeled_loss(mixed_seg_final, mixed_label, pixelWiseWeight)
        # # print(L_u)

        loss = L_1 + 0.25 * L_u

        # else :           
        # loss = L_1

        loss.backward()
        optimizer_fea.step()
        optimizer_cls.step()

        if ema_feature_extractor is not None:
            alpha_teacher = 0.999 
            ema_model = update_ema_variables(ema_feature_extractor, ema_classifier, feature_extractor,classifier, alpha_teacher, iteration)


        meters.update(loss_seg=loss.item())
        iteration += 1

        batch_time = time.time() - end
        end = time.time()

        wandb.log({
            "loss": loss.item(),
            "lr": current_lr,
            "batch_time": batch_time,
        })

        meters.update(time=batch_time, data=data_time)
        eta_seconds = meters.time.global_avg * (cfg.SOLVER.STOP_ITER - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        if iteration % 20 == 0 or iteration == max_iters:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.2f} GB",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer_fea.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 / 1024.0,
                )
            )

        if (iteration % cfg.SOLVER.CHECKPOINT_PERIOD == 0 or iteration == cfg.SOLVER.STOP_ITER):
            mIoU_final, mAcc_final, allAcc_final = run_test(cfg, feature_extractor, classifier, local_rank, distributed, logger)
            
            feature_extractor['B'].train()
            # feature_extractor['D'].train()
            # feature_extractor['S'].train()
            # feature_extractor['T'].train()
            # feature_extractor['FS'].train()
            classifier['S'].train()
            # classifier['D'].train()
            # classifier['T'].train()
            # classifier['FS'].train()

            if save_to_disk:
                # update best model
                if mIoU_final > best_mIoU:
                    filename = os.path.join(output_dir, "model_best.pth")
                    torch.save({'iteration': iteration, 'feature_extractor': feature_extractor.state_dict(),
                        'classifier': classifier.state_dict(), 'ema_feature_extractor': ema_feature_extractor.state_dict(),
                        'ema_classifier': ema_classifier.state_dict(), 'optimizer_fea': optimizer_fea.state_dict(),
                        'optimizer_cls': optimizer_cls.state_dict()}, filename)
                    best_mIoU = mIoU_final
                    best_iteration = iteration
                else:
                    filename = os.path.join(output_dir, "model_current.pth")
                    torch.save({'iteration': iteration, 'feature_extractor': feature_extractor.state_dict(),
                        'classifier': classifier.state_dict(), 'optimizer_fea': optimizer_fea.state_dict(),
                        'optimizer_cls': optimizer_cls.state_dict()}, filename)
                logger.info(f"-------- Best mIoU {best_mIoU} at iteration {best_iteration} --------")

            # Wandb Evaluation Log
            wandb.log({
                "Current Best mIoU": best_mIoU,
                "allAcc_final": allAcc_final,
                "mAcc_final": mAcc_final,
                "mIoU_final": mIoU_final,
            })
            
            torch.cuda.empty_cache()

        if iteration == cfg.SOLVER.MAX_ITER:
            break
        if iteration == cfg.SOLVER.STOP_ITER:
            break

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))

    wandb.finish()

    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / cfg.SOLVER.STOP_ITER
        )
    )

def run_test(cfg, feature_extractor, classifier, local_rank, distributed, logger):
    if local_rank == 0:
        logger.info('>>>>>>>>>>>>>>>> Start Testing >>>>>>>>>>>>>>>>')

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()

    seg_intersection_meter = AverageMeter()
    seg_union_meter = AverageMeter()
    seg_target_meter = AverageMeter()

    trans_intersection_meter = AverageMeter()
    trans_union_meter = AverageMeter()
    trans_target_meter = AverageMeter()

    final_intersection_meter = AverageMeter()
    final_union_meter = AverageMeter()
    final_target_meter = AverageMeter()

    if distributed:
        feature_extractor, classifier = feature_extractor.module, classifier.module
    torch.cuda.empty_cache()
    dataset_name = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        mkdir(output_folder)

    test_data = build_dataset(cfg, mode='test', is_source=False)
    if distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
    else:
        test_sampler = None
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=4,
                                              pin_memory=True, sampler=test_sampler)
    
    feature_extractor['B'].eval()
    # feature_extractor['D'].eval()
    # feature_extractor['S'].eval()
    # feature_extractor['T'].eval()
    # feature_extractor['FS'].eval()
    classifier['S'].eval()
    # classifier['D'].eval()
    # classifier['T'].eval()
    # classifier['FS'].eval()

    end = time.time()
    with torch.no_grad():
        for i, (x, y, _, _) in enumerate(test_loader):
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True).long()
            
            size = y.shape[-2:]

            src_feats_back = feature_extractor['B'](x)
            # src_feats_seg = feature_extractor['S'](src_feats_back)
            # src_feats_depth = feature_extractor['D'](src_feats_back)
            # src_feats_trans = feature_extractor['T'](src_feats_depth)
            # src_feats_final = feature_extractor['FS'](src_feats_seg, src_feats_trans)

            # output_seg = classifier['S'](src_feats_seg)
            # output_trans = classifier['T'](src_feats_trans)
            output_final = classifier['S'](src_feats_back)

            # output_seg = F.interpolate(output_seg, size=size, mode='bilinear', align_corners=True)
            # output_seg = output_seg.max(1)[1]
            # output_trans = F.interpolate(output_trans, size=size, mode='bilinear', align_corners=True)
            # output_trans = output_trans.max(1)[1]
            output_final = F.interpolate(output_final, size=size, mode='bilinear', align_corners=True)
            output_final = output_final.max(1)[1]
            
            # seg_intersection, seg_union, seg_target = intersectionAndUnionGPU(output_seg, y, cfg.MODEL.NUM_CLASSES,
            #                                                       cfg.INPUT.IGNORE_LABEL)
            # trans_intersection, trans_union, trans_target = intersectionAndUnionGPU(output_trans, y, cfg.MODEL.NUM_CLASSES,
            #                                                       cfg.INPUT.IGNORE_LABEL)
            final_intersection, final_union, final_target = intersectionAndUnionGPU(output_final, y, cfg.MODEL.NUM_CLASSES,
                                                                  cfg.INPUT.IGNORE_LABEL)
            # if distributed:
            #     torch.distributed.all_reduce(intersection), torch.distributed.all_reduce(
            #         union), torch.distributed.all_reduce(target)
                
            # seg_intersection, seg_union, seg_target = seg_intersection.cpu().numpy(), seg_union.cpu().numpy(), seg_target.cpu().numpy()
            # trans_intersection, trans_union, trans_target = trans_intersection.cpu().numpy(), trans_union.cpu().numpy(), trans_target.cpu().numpy()
            final_intersection, final_union, final_target = final_intersection.cpu().numpy(), final_union.cpu().numpy(), final_target.cpu().numpy()
            
            # seg_intersection_meter.update(seg_intersection), seg_union_meter.update(seg_union), seg_target_meter.update(seg_target)
            # trans_intersection_meter.update(trans_intersection), trans_union_meter.update(trans_union), trans_target_meter.update(trans_target)
            final_intersection_meter.update(final_intersection), final_union_meter.update(final_union), final_target_meter.update(final_target)

            # seg_accuracy = sum(seg_intersection_meter.val) / (sum(seg_target_meter.val) + 1e-10)
            # trans_accuracy = sum(trans_intersection_meter.val) / (sum(trans_target_meter.val) + 1e-10)
            # final_accuracy = sum(final_intersection_meter.val) / (sum(final_target_meter.val) + 1e-10)
            batch_time.update(time.time() - end)
            end = time.time()

    # iou_class_seg = seg_intersection_meter.sum / (seg_union_meter.sum + 1e-10)
    # accuracy_class_seg = seg_intersection_meter.sum / (seg_target_meter.sum + 1e-10)
    # mIoU_seg = np.mean(iou_class_seg)
    # mAcc_seg = np.mean(accuracy_class_seg)
    # allAcc_seg = sum(seg_intersection_meter.sum) / (sum(seg_target_meter.sum) + 1e-10)

    # iou_class_trans = trans_intersection_meter.sum / (trans_union_meter.sum + 1e-10)
    # accuracy_class_trans = trans_intersection_meter.sum / (trans_target_meter.sum + 1e-10)
    # mIoU_trans = np.mean(iou_class_trans)
    # mAcc_trans = np.mean(accuracy_class_trans)
    # allAcc_trans = sum(trans_intersection_meter.sum) / (sum(trans_target_meter.sum) + 1e-10)

    iou_class_final = final_intersection_meter.sum / (final_union_meter.sum + 1e-10)
    accuracy_class_final = final_intersection_meter.sum / (final_target_meter.sum + 1e-10)
    mIoU_final = np.mean(iou_class_final)
    mAcc_final = np.mean(accuracy_class_final)
    allAcc_final = sum(final_intersection_meter.sum) / (sum(final_target_meter.sum) + 1e-10)
    
    if local_rank == 0:
        # logger.info("Val seg result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}".format(mIoU_seg, mAcc_seg, allAcc_seg))
        # logger.info("Val trans result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}".format(mIoU_trans, mAcc_trans, allAcc_trans))
        logger.info("Val final result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}".format(mIoU_final, mAcc_final, allAcc_final))
        for i in range(cfg.MODEL.NUM_CLASSES):
            logger.info(
                "Class_{} {} Result: iou/accuracy {:.4f}/{:.4f}.".format(i, test_data.trainid2name[i],
                                                                         iou_class_final[i], accuracy_class_final[i])
            )
    return mIoU_final, mAcc_final, allAcc_final

def main():
    parser = argparse.ArgumentParser(description="PyTorch Semantic Segmentation Training")
    parser.add_argument("-cfg",
                        "--config-file",
                        default="",
                        metavar="FILE",
                        help="path to config file",
                        type=str,
                        )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("SelfSupervised", output_dir, args.local_rank)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))

    train(cfg, args.local_rank, args.distributed, logger)


if __name__ == "__main__":
    main()
