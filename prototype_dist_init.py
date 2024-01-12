import argparse
import logging
from collections import OrderedDict
import  time
import datetime

import torch
import torch.nn.functional as F
import torch.utils.data
# import torch.distributed
import torch.backends.cudnn

from core.configs import cfg
from core.datasets import build_dataset
from core.models import build_feature_extractor, build_classifier
from core.utils.misc import mkdir, AverageMeter
from core.utils.logger import setup_logger
from core.utils.metric_logger import MetricLogger

from core.utils.prototype_dist_estimator import prototype_dist_estimator

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


def prototype_dist_init(cfg, logger):
    _, backbone_name = cfg.MODEL.NAME.split('_')
    feature_num = 2048 if backbone_name.startswith('resnet') else 1024

    feat_seg_estimator = prototype_dist_estimator(feature_num=feature_num, ptype='S', cfg=cfg)
    feat_dep_estimator = prototype_dist_estimator(feature_num=feature_num, ptype='D', cfg=cfg)
    merged_estimator = prototype_dist_estimator(feature_num=feature_num*2, cfg=cfg)

    out_estimator = prototype_dist_estimator(feature_num=cfg.MODEL.NUM_CLASSES, cfg=cfg)

    device = torch.device(cfg.MODEL.DEVICE)

    NUM_CLASSES = {"S": 16, "D": 1}
    TASKNAMES = ["S", "D"]
    
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

    torch.cuda.empty_cache()

    # load checkpoint
    if cfg.resume:
        logger.info("Loading checkpoint from {}".format(cfg.resume))
        checkpoint = torch.load(cfg.resume, map_location=torch.device('cpu'))

        feature_extractor_weights = strip_prefix_if_present(checkpoint['feature_extractor'], 'module.')
        feature_extractor.load_state_dict(feature_extractor_weights)
        
        classifier_weights = strip_prefix_if_present(checkpoint['classifier'], 'module.')
        classifier.load_state_dict(classifier_weights)

    src_train_data = build_dataset(cfg, mode='train', is_source=True, epochwise=True)
    src_train_loader = torch.utils.data.DataLoader(src_train_data,
                                                   batch_size=cfg.SOLVER.BATCH_SIZE_VAL,
                                                   shuffle=False,
                                                   num_workers=4,
                                                   pin_memory=True,
                                                   drop_last=False)
    iteration = 0

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
    start_time = time.time()
    max_iters = len(src_train_loader)
    meters = MetricLogger(delimiter="  ")
    logger.info(">>>>>>>>>>>>>>>> Initialize prototype >>>>>>>>>>>>>>>>")
    logger.info(max_iters)

    with torch.no_grad():
        for i, (src_input, src_seg_label, src_dep_label, _) in enumerate(src_train_loader):
            data_time = time.time() - end

            src_input = src_input.cuda(non_blocking=True)
            src_seg_label = src_seg_label.cuda(non_blocking=True).long()
            src_dep_label = src_dep_label.cuda(non_blocking=True).long()

            size = src_seg_label.shape[-2:]

            src_feats_back = feature_extractor['B'](src_input)
            src_feats_seg = feature_extractor['S'](src_feats_back)
            src_feats_depth = feature_extractor['D'](src_feats_back)
            src_feats_trans = feature_extractor['T'](src_feats_depth)
            src_feats_final = feature_extractor['FS'](src_feats_seg, src_feats_trans)


            # src_out_depth = classifier['D'](src_feats_depth)
            # src_out_seg = classifier['S'](src_feats_seg)
            # src_out_trans = classifier['T'](src_feats_trans)
            src_out_final_seg = classifier['FS'](src_feats_final) 

            B, N, Hs, Ws = src_feats_seg.size()
            _, C, _, _ = src_out_final_seg.size()

            # source mask: downsample the ground-truth label
            src_mask = F.interpolate(src_seg_label.unsqueeze(0).float(), size=(Hs, Ws), mode='nearest').squeeze(0).long()
            src_mask = src_mask.contiguous().view(B * Hs * Ws, )

            # feature level
            src_feats_seg = src_feats_seg.permute(0, 2, 3, 1).contiguous().view(B * Hs * Ws, N)
            src_feats_trans = src_feats_trans.permute(0, 2, 3, 1).contiguous().view(B * Hs * Ws, N)
            feat_seg_estimator.update(features=src_feats_seg.detach().clone(), labels=src_mask)
            feat_dep_estimator.update(features=src_feats_trans.detach().clone(), labels=src_mask)
            merged_estimator.merge_update(proto1=feat_dep_estimator.Proto, 
                                          proto2=feat_seg_estimator.Proto, 
                                          amount=feat_dep_estimator.Amount)
            # output level
            src_out_final_seg = src_out_final_seg.permute(0, 2, 3, 1).contiguous().view(B * Hs * Ws, C)
            out_estimator.update(features=src_out_final_seg.detach().clone(), labels=src_mask)

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            iteration = iteration + 1
            eta_seconds = meters.time.global_avg * (max_iters - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % 20 == 0 or iteration == max_iters:
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "iter: {iter}",
                            "{meters}",
                            "max mem: {memory:.02f}"
                        ]
                    ).format(
                        eta=eta_string,
                        iter=iteration,
                        meters=str(meters),
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 / 1024.0
                    )
                )

            if iteration == max_iters:
                feat_seg_estimator.save(name='prototype_seg_dist.pth')
                feat_dep_estimator.save(name='prototype_dep_dist.pth')
                merged_estimator.save(name='prototype_merged_dist.pth')
                out_estimator.save(name='prototype_out_dist.pth')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_time / max_iters
        )
    )


def main():
    parser = argparse.ArgumentParser(description="PyTorch Semantic Segmentation Training")
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

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("prototype_dist_init", output_dir, 0)
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))

    prototype_dist_init(cfg, logger)


if __name__ == "__main__":
    main()
