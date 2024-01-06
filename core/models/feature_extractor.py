import torch
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from .layers import FrozenBatchNorm2d
from . import vgg, resnet

affine_par = True

NUM_OUTPUT = {"S": 16, "D": 1}
TASKNAMES = ["S", "D"]

class vgg_feature_extractor(nn.Module):
    def __init__(self, backbone_name, pretrained_weights=None, aux=False, pretrained_backbone=True, freeze_bn=False):
        super(vgg_feature_extractor, self).__init__()
        backbone = vgg.__dict__[backbone_name](
            pretrained=pretrained_backbone, pretrained_weights=pretrained_weights)

        features, _ = list(backbone.features.children()), list(backbone.classifier.children())

        # remove pool4/pool5
        features = nn.Sequential(*(features[i] for i in list(range(23)) + list(range(24, 30))))
        for i in [23, 25, 27]:
            features[i].dilation = (2, 2)
            features[i].padding = (2, 2)
        fc6 = nn.Conv2d(512, 1024, kernel_size=3, padding=4, dilation=4)
        fc7 = nn.Conv2d(1024, 1024, kernel_size=3, padding=4, dilation=4)

        backbone = nn.Sequential(
            *([features[i] for i in range(len(features))] + [fc6, nn.ReLU(inplace=True), fc7, nn.ReLU(inplace=True)]))
        return_layers = {'4': 'low_fea', '32': 'out'}

        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, x):
        feas = self.backbone(x)
        out = feas['out']
        return out

class BottleneckPad(nn.Module):
    expansion = 4

    def __init__(self, input_channels, intermediate_channels, stride=1, dilation=1, downsample=None):
        super(BottleneckPad, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, intermediate_channels, kernel_size=1, stride=stride, bias=False) # change
        self.bn1 = nn.BatchNorm2d(intermediate_channels,affine = affine_par)

        padding = dilation
        self.conv2 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=1, # change
                               padding=padding, bias=False, dilation = dilation)
        self.bn2 = nn.BatchNorm2d(intermediate_channels,affine = affine_par)
        self.conv3 = nn.Conv2d(intermediate_channels, intermediate_channels * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(intermediate_channels * 4, affine = affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class TransferBlock(nn.Module):
    expansion = 4

    def __init__(self, input_channels, intermediate_channels, stride=1, dilation=1, downsample=None):
        super(TransferBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, intermediate_channels, kernel_size=1, stride=stride, bias=False) # change
        self.bn1 = nn.BatchNorm2d(intermediate_channels,affine = affine_par)

        padding = dilation
        self.conv2 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=1, # change
                               padding=padding, bias=False, dilation = dilation)
        self.bn2 = nn.BatchNorm2d(intermediate_channels,affine = affine_par)
        self.conv3 = nn.Conv2d(intermediate_channels, intermediate_channels * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(intermediate_channels * 4, affine = affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride


    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        return out




class resnet_feature_extractor(nn.Module):
    def __init__(self, backbone_name, pretrained_weights=None, aux=False, pretrained_backbone=True, freeze_bn=False):
        super(resnet_feature_extractor, self).__init__()
        bn_layer = nn.BatchNorm2d
        if freeze_bn:
            bn_layer = FrozenBatchNorm2d
        backbone = resnet.__dict__[backbone_name](
            pretrained=pretrained_backbone,
            replace_stride_with_dilation=[False, True, True], pretrained_weights=pretrained_weights,
            norm_layer=bn_layer)
        return_layers = {'layer4': 'out'}
        if aux:
            return_layers['layer3'] = 'aux'
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
        
        # self.tasks = TASKNAMES
        # self.auxilary_tasks = TASKNAMES
        # self.channels = 2048
        # # Task-specific heads for initial prediction
        # self.task_prediction_heads = TaskPredictionModule(self.auxilary_tasks, self.channels)
        # self.transfer = Transfer(self.channels )
        # self.final_prediction = FinalPredictionModule(self.channels)
        
    def forward(self, x):
        out = self.backbone(x)['out']
        # out = self.task_prediction_heads(out)
        # out1 = self.transfer(out['D'])
        # output = {}
        # output['T'] = out1
        # for task in self.auxilary_tasks:
        #     output[task] = out[task]
        # #
        # output['FS'] =  self.final_prediction(output)
        return out

    def optim_parameters(self, learning_rate):
        return [{'params': self.backbone.parameters(), 'lr': learning_rate}]
                # {'params': self.task_prediction_heads.parameters(), 'lr': 10*learning_rate},
                # {'params': self.transfer.parameters(), 'lr': 10*learning_rate},
                # {'params': self.final_prediction.parameters(), 'lr': 10*learning_rate}]
    
    def optim_depth_parameters(self, learning_rate):
        return [{'params': self.backbone.parameters(), 'lr': learning_rate},
                {'params': self.task_prediction_heads.parameters(), 'lr': 10*learning_rate}]
 
class resnet_feature_extractor_v2(nn.Module):
    _batchnorm = (nn.BatchNorm2d, nn.SyncBatchNorm, nn.GroupNorm)
    def __init__(self, backbone_name, pretrained_weights=None, aux=False, pretrained_backbone=True, eval_bn=False):
        super(resnet_feature_extractor_v2, self).__init__()
        bn_layer = nn.BatchNorm2d
        self.eval_bn = eval_bn
        backbone = resnet.__dict__[backbone_name](
            pretrained=pretrained_backbone,
            replace_stride_with_dilation=[False, True, True], pretrained_weights=pretrained_weights,
            norm_layer=bn_layer)
        return_layers = {'layer4': 'out'}
        if aux:
            return_layers['layer3'] = 'aux'
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, x):
        out = self.backbone(x)['out']
        return out

    def train(self, mode=True):
        super().train(mode)
        if self.eval_bn:
            for m in self.modules():
                if isinstance(m, resnet_feature_extractor_v2._batchnorm):
                    m.eval()

# class TaskPredictionModule(nn.Module):
#     """
#         Make the initial task predictions from the backbone features.
#     """
#     def __init__(self, tasks, input_channels, intermediate_channels=2048):
#         super(TaskPredictionModule, self).__init__() 
#         self.tasks = tasks 
#         layers = {}
#         # conv_out = {}
        
#         for task in self.tasks:
#             if input_channels != intermediate_channels:
#                 downsample = nn.Sequential(nn.Conv2d(input_channels, intermediate_channels, kernel_size=1,
#                                                     stride=1, bias=False), nn.BatchNorm2d(intermediate_channels))
#             else:
#                 downsample = None
#             bottleneck1 = BottleneckPad(input_channels, intermediate_channels//4, downsample=downsample)
#             bottleneck2 = BottleneckPad(intermediate_channels, intermediate_channels//4, downsample=None)
#             # conv_out_ = nn.Conv2d(intermediate_channels, NUM_OUTPUT[task], 1)
#             layers[task] = nn.Sequential(bottleneck1, bottleneck2)
#             # conv_out[task] = conv_out_
        
#         self.layers = nn.ModuleDict(layers)
#         # self.conv_out = nn.ModuleDict(conv_out)


#     def forward(self, x):
#         out = {}
#         for task in self.tasks:
#             out[task] = self.layers[task](x)
#             # out[task] = self.conv_out[task](out[task])

#         return out 
                    
class TaskPredictionModule(nn.Module):
    """
        Make the initial task predictions from the backbone features.
    """
    def __init__(self, tasks, input_channels, intermediate_channels=2048):
        super(TaskPredictionModule, self).__init__() 
        
        if input_channels != intermediate_channels:
            downsample = nn.Sequential(nn.Conv2d(input_channels, intermediate_channels, kernel_size=1,
                                                stride=1, bias=False), nn.BatchNorm2d(intermediate_channels))
        else:
            downsample = None
        bottleneck1 = BottleneckPad(input_channels, intermediate_channels//4, downsample=downsample)
        bottleneck2 = BottleneckPad(intermediate_channels, intermediate_channels//4, downsample=None)
        
        self.layers = nn.Sequential(bottleneck1, bottleneck2)
    

    def forward(self, x):
        out = self.layers(x)

        return out 

class Transfer(nn.Module):
    def __init__(self, input_channels = 2048, intermediate_channels = 2048, stride=1, dilation=1, num_blocks=5):
        super(Transfer, self).__init__()
        layers = []

        if input_channels != intermediate_channels:
                downsample = nn.Sequential(nn.Conv2d(input_channels, intermediate_channels, kernel_size=1,
                                                    stride=1, bias=False), nn.BatchNorm2d(intermediate_channels))
        else:
            downsample = None
        layers.append(TransferBlock(input_channels, intermediate_channels//4, downsample = downsample))
        for i in range(num_blocks-2):
            layers.append(TransferBlock(intermediate_channels, intermediate_channels//4, downsample = None))
        layers.append(TransferBlock(intermediate_channels, intermediate_channels//4, downsample = None))
        self.transfer = nn.Sequential(*layers)
        # self.conv_out = nn.Conv2d(intermediate_channels, NUM_OUTPUT['S'], 1)

    def forward(self, x):
        out = self.transfer(x)
        # out = self.conv_out(out)
        return out


class SABlock(nn.Module):
    """ Spatial self-attention block """
    def __init__(self, in_channels, out_channels):
        super(SABlock, self).__init__()
        self.attention = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                                        nn.ReLU(inplace=True))
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)

    def forward(self, x):
        attention_mask = self.attention(x)
        features = self.conv(x)
        return torch.mul(features, attention_mask)

class FinalPredictionModule(nn.Module):
    """
        Perform Multi-Task Distillation
        We apply an attention mask to features from other tasks and
        add the result as a residual.
    """
    def __init__(self, channels):
        super(FinalPredictionModule, self).__init__()

        self.self_attention = SABlock(channels, channels)


    def forward(self, x, t):
        adapters = self.self_attention(t)
        # out = x['S'] + torch.sum(torch.stack([v for v in adapters]), dim=0)
        out = x['S'] + adapters
        # out = x['S'] + torch.stack([v for v in adapters])
        return out
    
