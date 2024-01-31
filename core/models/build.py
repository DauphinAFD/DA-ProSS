import logging
import torch
from . import resnet, vgg
from .feature_extractor import vgg_feature_extractor, resnet_feature_extractor, resnet_feature_extractor_v2, TaskPredictionModule, Transfer, FinalPredictionModule 
from .classifier import ASPP_Classifier_V2
from .discriminator import *


def build_feature_extractor(cfg, TASKNAMES):
    model_name, backbone_name = cfg.MODEL.NAME.split('_')
    if backbone_name.startswith('resnetv2'):
        backbone = resnet_feature_extractor_v2(backbone_name.replace("v2", ""), pretrained_weights=cfg.MODEL.WEIGHTS, aux=False,
                                            pretrained_backbone=True, eval_bn=cfg.MODEL.EVAL_BN)
    elif backbone_name.startswith('vgg'):
        backbone = vgg_feature_extractor(backbone_name, pretrained_weights=cfg.MODEL.WEIGHTS, aux=False,
                                         pretrained_backbone=True, freeze_bn=cfg.MODEL.FREEZE_BN)
    elif backbone_name.startswith('resnet'):

        heads = {}
        auxilary_tasks = TASKNAMES
        channels = 2048
        heads['B'] = resnet_feature_extractor(backbone_name, pretrained_weights=cfg.MODEL.WEIGHTS, aux=False,
                                            pretrained_backbone=True, freeze_bn=cfg.MODEL.FREEZE_BN)
        for task in TASKNAMES:
            heads[task] = TaskPredictionModule(auxilary_tasks, channels)
        heads['T'] = Transfer(1)
        heads['FS'] = FinalPredictionModule(channels)
        feature = nn.ModuleDict(heads)
    
    
    else:
        raise NotImplementedError
    return feature


def build_classifier(cfg, TASKNAMES, NUM_CLASSES):
    _, backbone_name = cfg.MODEL.NAME.split('_')
    if backbone_name.startswith('vgg'):
        heads = {}
        for task in TASKNAMES:
            heads[task] = ASPP_Classifier_V2(1024, [6, 12, 18, 24], [6, 12, 18, 24], NUM_CLASSES[task])
        classifier = nn.ModuleDict(heads)
        
    elif backbone_name.startswith('resnet'):
        heads = {}
        for task in TASKNAMES:
            heads[task] = ASPP_Classifier_V2(2048, [6, 12, 18, 24], [6, 12, 18, 24], NUM_CLASSES[task])
        heads['T'] = ASPP_Classifier_V2(2048, [6, 12, 18, 24], [6, 12, 18, 24], NUM_CLASSES['S']) 
        heads['FS'] = ASPP_Classifier_V2(4096, [6, 12, 18, 24], [6, 12, 18, 24], NUM_CLASSES['S']) 
        classifier = nn.ModuleDict(heads)
    else:
        raise NotImplementedError
    return classifier



def build_adversarial_discriminator(cfg, num_features=None, mid_nc=256):
    _, backbone_name = cfg.MODEL.NAME.split('_')
    if backbone_name.startswith('vgg'):
        if num_features is None:
            num_features = 1024
        model_D = PixelDiscriminator(num_features, mid_nc, num_classes=cfg.MODEL.NUM_CLASSES)
    elif backbone_name.startswith('resnet'):
        if num_features is None:
            num_features = 2048
        model_D = PixelDiscriminator(num_features, mid_nc, num_classes=cfg.MODEL.NUM_CLASSES)
    else:
        raise NotImplementedError
    return model_D
