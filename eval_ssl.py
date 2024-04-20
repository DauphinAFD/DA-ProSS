import argparse
import os
import datetime
import logging
import time
import numpy as np
import wandb
from PIL import Image
import torch
import torch.nn.functional as F
import torch.utils.data

from core.configs import cfg
from core.datasets import build_dataset
from core.models import build_feature_extractor, build_classifier
from core.solver import adjust_learning_rate
from core.utils.misc import mkdir, AverageMeter, intersectionAndUnionGPU, get_color_pallete
from core.utils.logger import setup_logger
from core.utils.metric_logger import MetricLogger
from core.utils.loss import BerhuLoss
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


def train(cfg, local_rank, distributed, logger):
    logger.info("Start training")

    NUM_CLASSES = {"S": 16, "D": 1}
    TASKNAMES = ["S", "D"]


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

    batch_size = cfg.SOLVER.BATCH_SIZE


    output_dir = cfg.OUTPUT_DIR
    

    save_to_disk = local_rank == 0

    iteration = 0

    assert cfg.resume
    if cfg.resume:
        logger.info("Loading checkpoint from {}".format(cfg.resume))
        checkpoint = torch.load(cfg.resume, map_location=torch.device('cpu'))
        model_weights = checkpoint['feature_extractor'] if distributed else strip_prefix_if_present(
            checkpoint['feature_extractor'], 'module.')
        feature_extractor.load_state_dict(model_weights)
        classifier_weights = checkpoint['classifier'] if distributed else strip_prefix_if_present(
            checkpoint['classifier'], 'module.')
        classifier.load_state_dict(classifier_weights)

    src_train_data = build_dataset(cfg, mode='train', is_source=True)

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(src_train_data)
    else:
        train_sampler = None


    ce_criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    berhu_criterion = BerhuLoss().cuda()

    max_iters = cfg.SOLVER.MAX_ITER
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")

    start_training_time = time.time()
    end = time.time()

    best_mIoU = 0
    best_iteration = 0

    batch_time = time.time() - end
    end = time.time()



    # if iteration % 20 == 0 or iteration == max_iters:
    #     logger.info(
    #         meters.delimiter.join(
    #             [
    #                 "eta: {eta}",
    #                 "iter: {iter}",
    #                 "{meters}",
    #                 "lr: {lr:.6f}",
    #                 "max mem: {memory:.2f} GB",
    #             ]
    #         ).format(
    #             eta=eta_string,
    #             iter=iteration,
    #             meters=str(meters),
    #             lr=optimizer_fea.param_groups[0]["lr"],
    #             memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 / 1024.0,
    #         )
    #     )

    mIoU_final, mAcc_final, allAscc_final = run_test(cfg, feature_extractor, classifier, local_rank, distributed, logger)
        
    torch.cuda.empty_cache()


    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))


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

    output_final_folder = os.path.join(cfg.OUTPUT_DIR, "final_output")
    mkdir(output_final_folder)

    output_seg_folder = os.path.join(cfg.OUTPUT_DIR, "seg_output")
    mkdir(output_seg_folder)

    output_trans_folder = os.path.join(cfg.OUTPUT_DIR, "trans_output")
    mkdir(output_trans_folder)

    output_depth_folder = os.path.join(cfg.OUTPUT_DIR, "depth_output")
    mkdir(output_depth_folder)

    output_gt_folder = os.path.join(cfg.OUTPUT_DIR, "gt")
    mkdir(output_gt_folder)

    end = time.time()
    with torch.no_grad():
        for i, (x, y, _, name) in enumerate(test_loader):
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
            output_depth = classifier['D'](src_feats_depth)

            print(output_seg.shape)
            print(size)
            output_seg = F.interpolate(output_seg, size=size, mode='bilinear', align_corners=True)
            output_seg = output_seg.max(1)[1]
            output_depth = F.interpolate(output_depth, size=size, mode='bilinear', align_corners=True)
            output_trans = F.interpolate(output_trans, size=size, mode='bilinear', align_corners=True)
            output_trans = output_trans.max(1)[1]
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

            mask_filename = name[0] if len(name[0].split("/")) < 2 else name[0].split("/")[-1]
            
            mask_seg = get_color_pallete(output_final[0].cpu().numpy(), "city")
            mask_seg.save(os.path.join(output_seg_folder, mask_filename))

            mask_trans = get_color_pallete(output_final[0].cpu().numpy(), "city")
            mask_trans.save(os.path.join(output_trans_folder, mask_filename))

            mask_trans = get_color_pallete(output_final[0].cpu().numpy(), "city")
            mask_trans.save(os.path.join(output_final_folder, mask_filename))

            mask_gt = get_color_pallete(y[0].cpu().numpy(), "city")
            mask_gt.save(os.path.join(output_gt_folder, mask_filename))

            min = output_depth.min()
            max = output_depth.max()
            max -= min
            depth_img = torch.round(((output_depth - min)/max) * 255.0)
            depth_out = Image.fromarray((depth_img.squeeze(0).squeeze(0)).cpu().numpy().astype(np.uint8))
            depth_out.save(os.path.join(output_depth_folder, mask_filename))
    
                
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
