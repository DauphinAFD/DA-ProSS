import argparse
import os
import datetime
import logging
import time
import numpy as np
from collections import OrderedDict
import wandb

import torch
import torch.distributed
import torch.utils.data
import torch.backends.cudnn
import torch.nn.functional as F
# from torchinfo import summary

from core.configs import cfg
from core.datasets import build_dataset
from core.models import build_feature_extractor, build_classifier
from core.solver import adjust_learning_rate
from core.utils.misc import mkdir, AverageMeter, intersectionAndUnionGPU
from core.utils.logger import setup_logger
from core.utils.metric_logger import MetricLogger
from core.utils.lovasz_loss import lovasz_softmax
from core.utils.loss import BerhuLoss


import warnings
warnings.filterwarnings('ignore')

def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith(prefix + 'layer5'):
            continue
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict


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
    feature_extractor['S'].to(device)
    feature_extractor['D'].to(device)
    feature_extractor['T'].to(device)
    feature_extractor['FS'].to(device)

    # classifier.to(device)
    classifier['S'].to(device)
    classifier['D'].to(device)
    classifier['T'].to(device)
    classifier['FS'].to(device)

    batch_size = cfg.SOLVER.BATCH_SIZE  # default: 8
    
    
    if local_rank == 0:
        logger.info(feature_extractor)
        logger.info(classifier)

    param_groups_fea = [{'params': feature_extractor['B'].parameters(), 'lr': cfg.SOLVER.BASE_LR},
                        {'params': feature_extractor['D'].parameters(), 'lr': cfg.SOLVER.BASE_LR},
                        {'params': feature_extractor['T'].parameters(), 'lr': 10*cfg.SOLVER.BASE_LR},
                        {'params': feature_extractor['S'].parameters(), 'lr': 10*cfg.SOLVER.BASE_LR},
                        {'params': feature_extractor['FS'].parameters(), 'lr': 10*cfg.SOLVER.BASE_LR}]
    
    optimizer_fea = torch.optim.SGD(param_groups_fea, momentum=cfg.SOLVER.MOMENTUM,
                                    weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer_fea.zero_grad()

    param_groups_cls = [{'params': classifier['D'].parameters(), 'lr': 1*cfg.SOLVER.BASE_LR},
                    {'params': classifier['T'].parameters(), 'lr': 10*cfg.SOLVER.BASE_LR},
                    {'params': classifier['S'].parameters(), 'lr': 10*cfg.SOLVER.BASE_LR},
                    {'params': classifier['FS'].parameters(), 'lr': 10*cfg.SOLVER.BASE_LR}]
    
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
        # if "optimizer_fea" in checkpoint:
        #     logger.info("Loading optimizer_fea from {}".format(cfg.resume))
        #     optimizer_fea.load(checkpoint['optimizer_fea'])
        # if "optimizer_cls" in checkpoint:
        #     logger.info("Loading optimizer_cls from {}".format(cfg.resume))
        #     optimizer_cls.load(checkpoint['optimizer_cls'])
        # if "iteration" in checkpoint:
        #     iteration = checkpoint['iteration']

    src_train_data = build_dataset(cfg, mode='train', is_source=True)

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(src_train_data)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(src_train_data,
                                               batch_size=batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=4,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               drop_last=True)

    ce_criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    berhu_criterion = BerhuLoss().cuda()

    max_iters = cfg.SOLVER.MAX_ITER
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")

    
    feature_extractor['B'].train()
    feature_extractor['D'].train()
    feature_extractor['S'].train()
    feature_extractor['T'].train()
    feature_extractor['FS'].train()
    
    classifier['S'].train()
    classifier['D'].train()
    classifier['T'].train()
    classifier['FS'].train()

    start_training_time = time.time()
    end = time.time()

    best_mIoU = 0
    best_iteration = 0

    for i, (src_input, src_seg_label, src_depth_label, _) in enumerate(train_loader):
        data_time = time.time() - end

        optimizer_fea.zero_grad()
        optimizer_cls.zero_grad()

        current_lr = adjust_learning_rate(cfg.SOLVER.LR_METHOD, cfg.SOLVER.BASE_LR, iteration, max_iters,
                                          power=cfg.SOLVER.LR_POWER)
        
        optimizer_fea.param_groups[0]['lr'] = current_lr
        optimizer_fea.param_groups[1]['lr'] = current_lr
        for index in range(2, len(optimizer_fea.param_groups)):
            optimizer_fea.param_groups[index]['lr'] = current_lr * 10

        optimizer_cls.param_groups[0]['lr'] = current_lr
        for index in range(1, len(optimizer_cls.param_groups)):
            optimizer_cls.param_groups[index]['lr'] = current_lr * 10

        

        src_input = src_input.cuda(non_blocking=True)
        src_seg_label = src_seg_label.cuda(non_blocking=True).long()
        src_depth_label = src_depth_label.cuda(non_blocking=True).long()

        size = src_seg_label.shape[-2:]

        src_feats_back = feature_extractor['B'](src_input)
        src_feats_seg = feature_extractor['S'](src_feats_back)
        src_feats_depth = feature_extractor['D'](src_feats_back)
        src_feats_trans = feature_extractor['T'](src_feats_depth)
        src_feats_final = feature_extractor['FS'](src_feats_seg, src_feats_trans)

        pred_seg = classifier['S'](src_feats_seg, size)
        pred_depth = classifier['D'](src_feats_depth, size)
        pred_trans = classifier['T'](src_feats_trans, size)
        pred_seg_final = classifier['FS'](src_feats_final, size)

        if cfg.SOLVER.LAMBDA_LOV > 0:
            pred_softmax = F.softmax(pred_seg, dim=1)
            pred_softmax_trans = F.softmax(pred_trans, dim=1)
            pred_softmax_final = F.softmax(pred_seg_final, dim=1)

            loss_lov_seg = lovasz_softmax(pred_softmax, src_seg_label, ignore=255)
            loss_lov_trans = lovasz_softmax(pred_softmax_trans, src_seg_label, ignore=255)
            loss_lov_final = lovasz_softmax(pred_softmax_final, src_seg_label, ignore=255)

            ce_seg = ce_criterion(pred_seg, src_seg_label)
            ce_trans = ce_criterion(pred_trans, src_seg_label)
            ce_final = ce_criterion(pred_seg_final, src_seg_label)
            loss_depth = berhu_criterion(pred_depth, src_depth_label)

            loss = ce_seg + ( cfg.SOLVER.LAMBDA_LOV * loss_lov_seg ) + \
                ce_trans + ( cfg.SOLVER.LAMBDA_LOV * loss_lov_trans ) + \
                    ce_final + ( cfg.SOLVER.LAMBDA_LOV * loss_lov_final ) + (cfg.SOLVER.LAMBDA_DEPTH * loss_depth) 
            
            meters.update(loss_lov=loss_lov_seg.item())
            meters.update(loss_lov_trans=loss_lov_trans.item())
            meters.update(loss_lov_final=loss_lov_final.item())
        else:
            loss = ce_seg + ce_trans + ce_final + (cfg.SOLVER.LAMBDA_DEPTH * loss_depth)

        loss.backward()

        optimizer_fea.step()
        optimizer_cls.step()
        meters.update(loss_seg=loss.item())
        iteration += 1

        batch_time = time.time() - end
        end = time.time()

        # Wandb Train Log
        wandb.log({
            "loss": loss.item(),
            "loss_ce_seg": ce_seg.item(),
            "loss_ce_trans": ce_trans.item(),
            "loss_ce_final": ce_final.item(),
            "loss_lov_seg": loss_lov_seg.item(),
            "loss_lov_trans": loss_lov_trans.item(),
            "loss_lov_final": loss_lov_final.item(),
            "loss_depth": loss_depth.item(),
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
            src_feats_trans = feature_extractor['T'](src_feats_depth)
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
    parser = argparse.ArgumentParser(description="PyTorch Semantic Segmentation Training")
    parser.add_argument("-cfg",
                        "--config-file",
                        default="",
                        metavar="FILE",
                        help="path to config file",
                        type=str,
                        )
    # parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    args.local_rank =  int(os.environ["LOCAL_RANK"])

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

    logger = setup_logger("SourceOnly", output_dir, args.local_rank)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))

    train(cfg, args.local_rank, args.distributed, logger)
    

if __name__ == "__main__":
    main()
