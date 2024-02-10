import argparse
import os
import datetime
import logging
import time
import numpy as np
from collections import OrderedDict
import wandb

import torch
import torch.utils
import torch.distributed
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn

from core.configs import cfg
from core.datasets import build_dataset
from core.models import build_feature_extractor, build_classifier
from core.solver import adjust_learning_rate
from core.utils.misc import mkdir, AverageMeter, intersectionAndUnionGPU
from core.utils.logger import setup_logger
from core.utils.metric_logger import MetricLogger
from core.utils.lovasz_loss import lovasz_softmax
from core.utils.prototype_dist_estimator import prototype_dist_estimator
from core.utils.loss import BerhuLoss, PrototypeContrastiveLoss

import warnings
warnings.filterwarnings('ignore')


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict


def train(cfg, local_rank, distributed, logger):
    
    # Initialize wandb
    run = wandb.init(
        project = 'DA-ProSS',
        config = cfg,
        # name = cfg.run_name
    )

    device = torch.device(cfg.MODEL.DEVICE)
    NUM_CLASSES = {"S": 16, "D": 1}
    TASKNAMES = ["S", "D"]

    # create network
    feature_extractor = build_feature_extractor(cfg, TASKNAMES)
    feature_extractor['B'].to(device)
    feature_extractor['S'].to(device)
    feature_extractor['D'].to(device)
    feature_extractor['T'].to(device)
    feature_extractor['FS'].to(device)

    classifier = build_classifier(cfg, TASKNAMES, NUM_CLASSES)
    classifier['S'].to(device)
    classifier['D'].to(device)
    classifier['T'].to(device)
    classifier['FS'].to(device)

    # batch size: half for source and half for target
    batch_size = cfg.SOLVER.BATCH_SIZE // 2

    logger.info(feature_extractor)
    logger.info(classifier)


    # init optimizers
    param_groups_fea = [{'params': feature_extractor['B'].parameters(), 'lr': cfg.SOLVER.BASE_LR},
                        {'params': feature_extractor['D'].parameters(), 'lr': 10*cfg.SOLVER.BASE_LR},
                        {'params': feature_extractor['T'].parameters(), 'lr': 10*cfg.SOLVER.BASE_LR},
                        {'params': feature_extractor['S'].parameters(), 'lr': 10*cfg.SOLVER.BASE_LR},
                        {'params': feature_extractor['FS'].parameters(), 'lr': 10*cfg.SOLVER.BASE_LR}]
    
    optimizer_fea = torch.optim.SGD(param_groups_fea, momentum=cfg.SOLVER.MOMENTUM,
                                    weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer_fea.zero_grad()

    param_groups_cls = [{'params': classifier['D'].parameters(), 'lr': 10*cfg.SOLVER.BASE_LR},
                    {'params': classifier['T'].parameters(), 'lr': 10*cfg.SOLVER.BASE_LR},
                    {'params': classifier['S'].parameters(), 'lr': 10*cfg.SOLVER.BASE_LR},
                    {'params': classifier['FS'].parameters(), 'lr': 10*cfg.SOLVER.BASE_LR}]
    
    optimizer_cls = torch.optim.SGD(param_groups_cls, momentum=cfg.SOLVER.MOMENTUM,
                                    weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer_cls.zero_grad()

    output_dir = cfg.OUTPUT_DIR

    if cfg.resume:
        logger.info("Loading checkpoint from {}".format(cfg.resume))
        checkpoint = torch.load(cfg.resume, map_location=torch.device('cpu'))
        model_weights = checkpoint['feature_extractor'] if distributed else strip_prefix_if_present(
            checkpoint['feature_extractor'], 'module.')
        feature_extractor.load_state_dict(model_weights)
        classifier_weights = checkpoint['classifier'] if distributed else strip_prefix_if_present(
            checkpoint['classifier'], 'module.')
        classifier.load_state_dict(classifier_weights)
        # if "optimizer_fea" in checkpoint:
        #     logger.info("Loading optimizer_fea from {}".format(cfg.resume))
        #     optimizer_fea.load(checkpoint['optimizer_fea'])
        # if "optimizer_cls" in checkpoint:
        #     logger.info("Loading optimizer_cls from {}".format(cfg.resume))
        #     optimizer_cls.load(checkpoint['optimizer_cls'])
        # if "iteration" in checkpoint:
        #     iteration = checkpoint['iteration']

    # init data loader
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


    # init loss
    ce_criterion = nn.CrossEntropyLoss(ignore_index=255)
    berhu_criterion = BerhuLoss().cuda()

    pcl_criterion = PrototypeContrastiveLoss(cfg)


    logger.info(">>>>>>>>>>>>>>>> Load init prototypes >>>>>>>>>>>>>>>>")
    _, backbone_name = cfg.MODEL.NAME.split('_')
    feature_num = 2048 if backbone_name.startswith('resnet') else 1024

    # Create prototypes
    feat_seg_estimator = prototype_dist_estimator(feature_num=feature_num, ptype='S', cfg=cfg)
    feat_dep_estimator = prototype_dist_estimator(feature_num=feature_num, ptype='D', cfg=cfg)
    merged_estimator = prototype_dist_estimator(feature_num=feature_num*2, cfg=cfg)

    if cfg.SOLVER.MULTI_LEVEL:
        out_seg_estimator = prototype_dist_estimator(feature_num=cfg.MODEL.NUM_CLASSES, ptype='S', cfg=cfg)
        out_trans_estimator = prototype_dist_estimator(feature_num=cfg.MODEL.NUM_CLASSES, ptype='T', cfg=cfg)
        out_final_estimator = prototype_dist_estimator(feature_num=cfg.MODEL.NUM_CLASSES, ptype='F', cfg=cfg)


    iteration = 0
    start_training_time = time.time()
    end = time.time()
    save_to_disk = local_rank == 0
    max_iters = cfg.SOLVER.MAX_ITER
    meters = MetricLogger(delimiter="  ")


    logger.info(">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>")

    feature_extractor['B'].train()
    feature_extractor['D'].train()
    feature_extractor['S'].train()
    feature_extractor['T'].train()
    feature_extractor['FS'].train()

    classifier['S'].train()
    classifier['D'].train()
    classifier['T'].train()
    classifier['FS'].train()

    best_mIoU = 0
    best_iteration = 0
    loss_out = 0


    # Modify data loader
    for i, ((src_input, src_seg_label, src_depth_label, src_name), (tgt_input, tgt_depth_label,_, _)) in enumerate(zip(src_train_loader, tgt_train_loader)):
        data_time = time.time() - end

        optimizer_fea.zero_grad()
        optimizer_cls.zero_grad()

        # Custom Leaning rate adjustment
        current_lr = adjust_learning_rate(cfg.SOLVER.LR_METHOD, cfg.SOLVER.BASE_LR, iteration, max_iters,
                                          power=cfg.SOLVER.LR_POWER)
        
        optimizer_fea.param_groups[0]['lr'] = current_lr
        for index in range(1, len(optimizer_fea.param_groups)):
            optimizer_fea.param_groups[index]['lr'] = current_lr * 10

        # optimizer_cls.param_groups[0]['lr'] = current_lr
        for index in range(len(optimizer_cls.param_groups)):
            optimizer_cls.param_groups[index]['lr'] = current_lr * 10



        # Initialize inputs
        src_input = src_input.cuda(non_blocking=True)
        src_depth_label = src_depth_label.cuda(non_blocking=True).long()
        src_seg_label = src_seg_label.cuda(non_blocking=True).long()
        tgt_input = tgt_input.cuda(non_blocking=True)
        tgt_depth_label = tgt_depth_label.cuda(non_blocking=True).long()
        
        src_size = src_input.shape[-2:]
        tgt_size = tgt_input.shape[-2:]

        # Source feature and classifier outputs
        src_feats_back = feature_extractor['B'](src_input)
        src_feats_seg = feature_extractor['S'](src_feats_back)
        src_feats_depth = feature_extractor['D'](src_feats_back)

        src_out_depth = classifier['D'](src_feats_depth)

        src_feats_trans = feature_extractor['T'](src_out_depth)
        src_feats_final = feature_extractor['FS'](src_feats_seg, src_feats_trans)

        src_out_seg = classifier['S'](src_feats_seg)
        src_out_trans = classifier['T'](src_feats_trans)
        src_out_final_seg = classifier['FS'](src_feats_final) 


        # Target feature and classifier outputs
        tgt_feats_back = feature_extractor['B'](tgt_input)
        tgt_feats_seg = feature_extractor['S'](tgt_feats_back)
        tgt_feats_depth = feature_extractor['D'](tgt_feats_back)

        tgt_out_depth = classifier['D'](tgt_feats_depth)

        tgt_feats_trans = feature_extractor['T'](tgt_out_depth)
        tgt_feats_final = feature_extractor['FS'](tgt_feats_seg, tgt_feats_trans)

        tgt_out_seg = classifier['S'](tgt_feats_seg)
        tgt_out_trans = classifier['T'](tgt_feats_trans)
        tgt_out_final_seg = classifier['FS'](tgt_feats_final)

        

# ----------------------------------------------------------------------------
        src_pred_seg = F.interpolate(src_out_seg, size=src_size, mode='bilinear', align_corners=True)
        src_pred_trans = F.interpolate(src_out_trans, size=src_size, mode='bilinear', align_corners=True)
        src_pred_final_seg = F.interpolate(src_out_final_seg, size=src_size, mode='bilinear', align_corners=True)

        src_pred_depth = F.interpolate(src_out_depth, size=src_size, mode='bilinear', align_corners=True)


        # Supervision loss
        if cfg.SOLVER.LAMBDA_LOV > 0:
            src_softmax_seg = F.softmax(src_pred_seg, dim=1)
            src_softmax_trans = F.softmax(src_pred_trans, dim=1)
            src_softmax_final = F.softmax(src_pred_final_seg, dim=1)

            loss_lov = lovasz_softmax(src_softmax_seg, src_seg_label, ignore=255)
            loss_lov_trans = lovasz_softmax(src_softmax_trans, src_seg_label, ignore=255)
            loss_lov_final = lovasz_softmax(src_softmax_final, src_seg_label, ignore=255)

            # Combining all the losses
            loss_sup = ce_criterion(src_pred_seg, src_seg_label) + ( cfg.SOLVER.LAMBDA_LOV * loss_lov ) + \
                    ce_criterion(src_pred_trans, src_seg_label) + ( cfg.SOLVER.LAMBDA_LOV * loss_lov_trans ) + \
                    ce_criterion(src_pred_final_seg, src_seg_label) + ( cfg.SOLVER.LAMBDA_LOV * loss_lov_final ) + \
                    (cfg.SOLVER.LAMBDA_DEPTH * berhu_criterion(src_pred_depth, src_depth_label)) 
                    
             
            meters.update(loss_lov=loss_lov.item())
            meters.update(loss_lov_trans=loss_lov_trans.item())
            meters.update(loss_lov_final=loss_lov_final.item())
        else:
            loss_sup = ce_criterion(src_out_seg, src_seg_label) + ce_criterion(src_out_final_seg, src_seg_label) \
                    + ce_criterion(src_out_trans, src_seg_label) + berhu_criterion(src_pred_depth,src_depth_label)


        # source mask: downsample the ground-truth label
        B, A, Hs, Ws = src_feats_seg.size()
        src_mask = F.interpolate(src_seg_label.unsqueeze(0).float(), size=(Hs, Ws), mode='nearest').squeeze(0).long()
        src_mask = src_mask.contiguous().view(B * Hs * Ws, )
        assert not src_mask.requires_grad

        # _, Af, Hfs, Wfs = src_feats_final.size()
        # final_src_mask = F.interpolate(src_seg_label.unsqueeze(0).float(), size=(Hfs, Wfs), mode='nearest').squeeze(0).long()
        # final_src_mask = final_src_mask.contiguous().view(B * Hfs * Wfs, )
        # assert not final_src_mask.requires_grad

        # target mask: constant threshold -- cfg.SOLVER.THRESHOLD
        _, _, Ht, Wt = tgt_feats_seg.size()
        tgt_out_maxvalue, tgt_final_mask = torch.max(tgt_out_final_seg, dim=1)
        for i in range(cfg.MODEL.NUM_CLASSES):
            tgt_final_mask[(tgt_out_maxvalue < cfg.SOLVER.DELTA) * (tgt_final_mask == i)] = 255
        tgt_final_mask = tgt_final_mask.contiguous().view(B * Ht * Wt, )
        assert not tgt_final_mask.requires_grad

        tgt_out_maxvalue, tgt_seg_mask = torch.max(tgt_out_seg, dim=1)
        for i in range(cfg.MODEL.NUM_CLASSES):
            tgt_seg_mask[(tgt_out_maxvalue < cfg.SOLVER.DELTA) * (tgt_seg_mask == i)] = 255
        tgt_seg_mask = tgt_seg_mask.contiguous().view(B * Ht * Wt, )
        assert not tgt_seg_mask.requires_grad

        tgt_out_maxvalue, tgt_trans_mask = torch.max(tgt_out_trans, dim=1)
        for i in range(cfg.MODEL.NUM_CLASSES):
            tgt_trans_mask[(tgt_out_maxvalue < cfg.SOLVER.DELTA) * (tgt_trans_mask == i)] = 255
        tgt_trans_mask = tgt_trans_mask.contiguous().view(B * Ht * Wt, )
        assert not tgt_trans_mask.requires_grad

        # _, _, Hft, Wft = tgt_feats_final.size()
        # final_tgt_out_maxvalue, final_tgt_mask = torch.max(tgt_out_final_seg, dim=1)
        # for i in range(cfg.MODEL.NUM_CLASSES):
        #     final_tgt_mask[(final_tgt_out_maxvalue < cfg.SOLVER.DELTA) * (final_tgt_mask == i)] = 255
        # final_tgt_mask = final_tgt_mask.contiguous().view(B * Hft * Wft, )
        # assert not final_tgt_mask.requires_grad

        
        src_feats_seg = src_feats_seg.permute(0, 2, 3, 1).contiguous().view(B * Hs * Ws, A)
        tgt_feats_seg = tgt_feats_seg.permute(0, 2, 3, 1).contiguous().view(B * Ht * Wt, A)

        src_feats_trans = src_feats_trans.permute(0, 2, 3, 1).contiguous().view(B * Hs * Ws, A)
        tgt_feats_trans = tgt_feats_trans.permute(0, 2, 3, 1).contiguous().view(B * Ht * Wt, A)

        # src_feats_final = src_feats_final.permute(0, 2, 3, 1).contiguous().view(B * Hs * Ws, 2*A)
        # tgt_feats_final = tgt_feats_final.permute(0, 2, 3, 1).contiguous().view(B * Ht * Wt, 2*A)


        # Update feature-level prototype statistics
        feat_dep_estimator.update(features=src_feats_trans.detach(), labels=src_mask)
        feat_dep_estimator.update(features=tgt_feats_trans.detach(), labels=tgt_trans_mask)

        feat_seg_estimator.update(features=src_feats_seg.detach(), labels=src_mask)
        feat_seg_estimator.update(features=tgt_feats_seg.detach(), labels=tgt_seg_mask)

        # merged_estimator.update(features=src_feats_final.detach(), labels=src_mask)
        # merged_estimator.update(features=tgt_feats_final.detach(), labels=tgt_final_mask)
        


        # contrastive loss on both domains
        loss_feat = pcl_criterion(Proto=feat_seg_estimator.Proto.detach(),
                                  feat=src_feats_seg,
                                  labels=src_mask) \
                    + pcl_criterion(Proto=feat_dep_estimator.Proto.detach(),
                                  feat=src_feats_trans,
                                  labels=src_mask) \
                    + pcl_criterion(Proto=feat_seg_estimator.Proto.detach(),
                                  feat=tgt_feats_seg,
                                  labels=tgt_seg_mask) \
                    + pcl_criterion(Proto=feat_dep_estimator.Proto.detach(),
                                  feat=tgt_feats_trans,
                                  labels=tgt_trans_mask) 
                    # + pcl_criterion(Proto=merged_estimator.Proto.detach(),
                    #               feat=tgt_feats_final,
                    #               labels=tgt_final_mask) \
                    # + pcl_criterion(Proto=merged_estimator.Proto.detach(),
                    #               feat=src_feats_final,
                    #               labels=src_mask)
        meters.update(loss_feat=loss_feat.item())       # Logger

        


        # Output level prototypes
        if cfg.SOLVER.MULTI_LEVEL:

            src_out_seg = src_out_seg.permute(0, 2, 3, 1).contiguous().view(B * Hs * Ws, cfg.MODEL.NUM_CLASSES)
            tgt_out_seg = tgt_out_seg.permute(0, 2, 3, 1).contiguous().view(B * Ht * Wt, cfg.MODEL.NUM_CLASSES)
            
            src_out_trans = src_out_trans.permute(0, 2, 3, 1).contiguous().view(B * Hs * Ws, cfg.MODEL.NUM_CLASSES)
            tgt_out_trans = tgt_out_trans.permute(0, 2, 3, 1).contiguous().view(B * Ht * Wt, cfg.MODEL.NUM_CLASSES)
            
            src_out_final_seg = src_out_final_seg.permute(0, 2, 3, 1).contiguous().view(B * Hs * Ws, cfg.MODEL.NUM_CLASSES)
            tgt_out_final_seg = tgt_out_final_seg.permute(0, 2, 3, 1).contiguous().view(B * Ht * Wt, cfg.MODEL.NUM_CLASSES)

            
            # update output-level statistics
            # out_seg_estimator.update(features=tgt_out_seg.detach(), labels=tgt_seg_mask)
            # out_seg_estimator.update(features=src_out_seg.detach(), labels=src_mask)

            # out_trans_estimator.update(features=tgt_out_trans.detach(), labels=tgt_trans_mask)
            out_trans_estimator.update(features=src_out_trans.detach(), labels=src_mask)
            
            out_final_estimator.update(features=tgt_out_final_seg.detach(), labels=tgt_final_mask)
            out_final_estimator.update(features=src_out_final_seg.detach(), labels=src_mask)

            
            # the proposed contrastive loss on prediction map
            # loss_out = pcl_criterion(Proto=out_seg_estimator.Proto.detach(),
            #                          feat=src_out_seg,
            #                          labels=src_mask) \
            #            + pcl_criterion(Proto=out_seg_estimator.Proto.detach(),
            #                            feat=tgt_out_seg,
            #                            labels=tgt_seg_mask) \
            #            + pcl_criterion(Proto=out_trans_estimator.Proto.detach(),
            #                          feat=src_out_trans,
            #                          labels=src_mask) \
            #            + pcl_criterion(Proto=out_trans_estimator.Proto.detach(),
            #                            feat=tgt_out_trans,
            #                            labels=tgt_trans_mask) \
            loss_out = pcl_criterion(Proto=out_final_estimator.Proto.detach(),
                                     feat=src_out_final_seg,
                                     labels=src_mask) \
                       + pcl_criterion(Proto=out_final_estimator.Proto.detach(),
                                       feat=tgt_out_final_seg,
                                       labels=tgt_final_mask)
            
            meters.update(loss_out=loss_out.item())

            loss = loss_sup \
                   + cfg.SOLVER.LAMBDA_FEAT * loss_feat \
                   + cfg.SOLVER.LAMBDA_OUT * loss_out
        else:
            loss = loss_sup + cfg.SOLVER.LAMBDA_FEAT * loss_feat



        loss.backward()

        optimizer_fea.step()
        optimizer_cls.step()


        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        # Wandb Train Log
        wandb.log({
            "loss": loss.item(),
            "loss_sup": loss_sup.item(),
            "loss_feat": loss_feat.item(),
            "loss_out": loss_out.item(),
            "lr": current_lr,
            "batch_time": batch_time,
        })

        eta_seconds = meters.time.global_avg * (cfg.SOLVER.STOP_ITER - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        iteration += 1
        if iteration % 20 == 0 or iteration == max_iters:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.02f} GB"
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer_fea.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 / 1024.0
                )
            )

        # Evaluate the model after iteration
        if (iteration % cfg.SOLVER.CHECKPOINT_PERIOD == 0 or iteration == cfg.SOLVER.STOP_ITER):
            mIoU_seg, mIoU_trans, mIoU_final = run_test(cfg, feature_extractor, classifier, local_rank, distributed, logger)

            feature_extractor['B'].train()
            feature_extractor['D'].train()
            feature_extractor['S'].train()
            feature_extractor['T'].train()
            feature_extractor['FS'].train()
            classifier['S'].train()
            classifier['D'].train()
            classifier['T'].train()
            classifier['FS'].train()

            if save_to_disk:
                # update best model
                if mIoU_final > best_mIoU:
                    filename = os.path.join(output_dir, "model_best.pth")
                    torch.save({'iteration': iteration, 'feature_extractor': feature_extractor.state_dict(),
                        'classifier': classifier.state_dict(), 'optimizer_fea': optimizer_fea.state_dict(),
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
                "mIoU_seg": mIoU_seg,
                "mIoU_trans": mIoU_trans,
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

    # return feature_extractor, classifier



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
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=0,
                                              pin_memory=True, sampler=test_sampler)
    
    feature_extractor['B'].eval()
    feature_extractor['D'].eval()
    feature_extractor['S'].eval()
    feature_extractor['T'].eval()
    feature_extractor['FS'].eval()
    classifier['S'].eval()
    classifier['D'].eval()
    classifier['T'].eval()
    classifier['FS'].eval()

    end = time.time()
    with torch.no_grad():
        for i, (x, y, _, _) in enumerate(test_loader):
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True).long()
            
            size = y.shape[-2:]

            src_feats_back = feature_extractor['B'](x)
            src_feats_seg = feature_extractor['S'](src_feats_back)
            src_feats_depth = feature_extractor['D'](src_feats_back)
            
            output_depth = classifier['D'](src_feats_depth)
            src_feats_trans = feature_extractor['T'](output_depth)
            src_feats_final = feature_extractor['FS'](src_feats_seg, src_feats_trans)

            output_seg = classifier['S'](src_feats_seg)
            output_trans = classifier['T'](src_feats_trans)
            output_final = classifier['FS'](src_feats_final)

            output_seg = F.interpolate(output_seg, size=size, mode='bilinear', align_corners=True)
            output_seg = output_seg.max(1)[1]
            output_trans = F.interpolate(output_trans, size=size, mode='bilinear', align_corners=True)
            output_trans = output_trans.max(1)[1]
            output_final = F.interpolate(output_final, size=size, mode='bilinear', align_corners=True)
            output_final = output_final.max(1)[1]
            
            seg_intersection, seg_union, seg_target = intersectionAndUnionGPU(output_seg, y, cfg.MODEL.NUM_CLASSES,
                                                                  cfg.INPUT.IGNORE_LABEL)
            trans_intersection, trans_union, trans_target = intersectionAndUnionGPU(output_trans, y, cfg.MODEL.NUM_CLASSES,
                                                                  cfg.INPUT.IGNORE_LABEL)
            final_intersection, final_union, final_target = intersectionAndUnionGPU(output_final, y, cfg.MODEL.NUM_CLASSES,
                                                                  cfg.INPUT.IGNORE_LABEL)
            # if distributed:
            #     torch.distributed.all_reduce(intersection), torch.distributed.all_reduce(
            #         union), torch.distributed.all_reduce(target)
                
            seg_intersection, seg_union, seg_target = seg_intersection.cpu().numpy(), seg_union.cpu().numpy(), seg_target.cpu().numpy()
            trans_intersection, trans_union, trans_target = trans_intersection.cpu().numpy(), trans_union.cpu().numpy(), trans_target.cpu().numpy()
            final_intersection, final_union, final_target = final_intersection.cpu().numpy(), final_union.cpu().numpy(), final_target.cpu().numpy()
            
            seg_intersection_meter.update(seg_intersection), seg_union_meter.update(seg_union), seg_target_meter.update(seg_target)
            trans_intersection_meter.update(trans_intersection), trans_union_meter.update(trans_union), trans_target_meter.update(trans_target)
            final_intersection_meter.update(final_intersection), final_union_meter.update(final_union), final_target_meter.update(seg_target)

            # seg_accuracy = sum(seg_intersection_meter.val) / (sum(seg_target_meter.val) + 1e-10)
            # trans_accuracy = sum(trans_intersection_meter.val) / (sum(trans_target_meter.val) + 1e-10)
            # final_accuracy = sum(final_intersection_meter.val) / (sum(final_target_meter.val) + 1e-10)
            batch_time.update(time.time() - end)
            end = time.time()

    iou_class_seg = seg_intersection_meter.sum / (seg_union_meter.sum + 1e-10)
    accuracy_class_seg = seg_intersection_meter.sum / (seg_target_meter.sum + 1e-10)
    mIoU_seg = np.mean(iou_class_seg)
    mAcc_seg = np.mean(accuracy_class_seg)
    allAcc_seg = sum(seg_intersection_meter.sum) / (sum(seg_target_meter.sum) + 1e-10)

    iou_class_trans = trans_intersection_meter.sum / (trans_union_meter.sum + 1e-10)
    accuracy_class_trans = trans_intersection_meter.sum / (trans_target_meter.sum + 1e-10)
    mIoU_trans = np.mean(iou_class_trans)
    mAcc_trans = np.mean(accuracy_class_trans)
    allAcc_trans = sum(trans_intersection_meter.sum) / (sum(trans_target_meter.sum) + 1e-10)

    iou_class_final = final_intersection_meter.sum / (final_union_meter.sum + 1e-10)
    accuracy_class_final = final_intersection_meter.sum / (final_target_meter.sum + 1e-10)
    mIoU_final = np.mean(iou_class_final)
    mAcc_final = np.mean(accuracy_class_final)
    allAcc_final = sum(final_intersection_meter.sum) / (sum(final_target_meter.sum) + 1e-10)
    
    if local_rank == 0:
        logger.info("Val seg result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}".format(mIoU_seg, mAcc_seg, allAcc_seg))
        logger.info("Val trans result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}".format(mIoU_trans, mAcc_trans, allAcc_trans))
        logger.info("Val final result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}".format(mIoU_final, mAcc_final, allAcc_final))
        for i in range(cfg.MODEL.NUM_CLASSES):
            logger.info(
                "Class_{} {} Result: iou/accuracy {:.4f}/{:.4f}.".format(i, test_data.trainid2name[i],
                                                                         iou_class_final[i], accuracy_class_final[i])
            )
    return mIoU_seg, mIoU_trans, mIoU_final


def main():
    parser = argparse.ArgumentParser(description="Pytorch Domain Adaptive Semantic Segmentation Training")
    parser.add_argument("-cfg",
                        "--config-file",
                        default="",
                        metavar="FILE",
                        help="path to config file",
                        type=str)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER
    )

    args = parser.parse_args()

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

    logger = setup_logger("ProCA", output_dir, args.local_rank)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))

    train(cfg, args.local_rank, args.distributed, logger)



if __name__ == '__main__':
    main()