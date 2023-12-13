import torch
from torch import nn
import torch.nn.functional as F

affine_par = True

NUM_OUTPUT = {"S": 16, "D": 1}
TASKNAMES = ["S", "D"]

class BasicBlockTranser(nn.Module):

    def __init__(self, input_channels, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlockTranser, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes,affine = affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(planes,affine = affine_par)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn3 = nn.BatchNorm2d(planes,affine = affine_par)
        self.downsample = downsample

    def forward(self, x):

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


class Transfer(nn.Module):
    def __init__(self, input_channels = 2048, intermediate_channels = 256, stride=1, dilation=1, num_blocks=5):
        super(Transfer, self).__init__()
        layers = []

        if input_channels != intermediate_channels:
                downsample = nn.Sequential(nn.Conv2d(input_channels, intermediate_channels, kernel_size=1,
                                                    stride=1, bias=False), nn.BatchNorm2d(intermediate_channels))
        else:
            downsample = None
        layers.append(BasicBlockTranser(input_channels, intermediate_channels, downsample = downsample))
        for i in range(num_blocks-2):
            layers.append(BasicBlockTranser(intermediate_channels, intermediate_channels))
        layers.append(BasicBlockTranser(intermediate_channels, intermediate_channels))
        self.transfer = nn.Sequential(*layers)
        self.conv_out = nn.Conv2d(intermediate_channels, NUM_OUTPUT['S'], 1)

    def forward(self, x):
        out = self.transfer(x)
        out = self.conv_out(out)

        return out
    



class AdaptiveNet(nn.Module):
    def __init__(self, encoder, transfer):
        super(AdaptiveNet, self).__init__()
        self.encoder = encoder
        self.transfer = transfer

    def forward(self, x):
        out = self.encoder(x)
        out = self.transfer(out)
        return out
