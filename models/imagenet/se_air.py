from __future__ import division
""" 
Ported form 
https://github.com/facebook/fb.resnet.torch
and
https://arxiv.org/pdf/1703.06211.pdf
(c) Yang Lu
"""
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch

__all__ = ['se_air26', 'se_air50', 'se_air101', 'se_air152']

class SEBottleneck(nn.Module):
    """
    SEBottleneck bottleneck
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, ave_kernel=56):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
        """
        super(SEBottleneck, self).__init__()	

        self.conv1_1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1_1 = nn.BatchNorm2d(planes)
        self.conv1_2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.conv2_1 = nn.Conv2d(inplanes, int(planes*0.5), kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2_1 = nn.BatchNorm2d(int(planes*0.5))
        self.conv2_2 = nn.Conv2d(int(planes*0.5), int(planes*0.5), kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(int(planes*0.5))
        self.conv2_3 = nn.Conv2d(int(planes*0.5),  int(planes*0.5), kernel_size=3, stride=1, padding=1, bias=False)

        self.bn_concat = nn.BatchNorm2d(int(planes*1.5))

        self.conv = nn.Conv2d(int(planes*1.5), planes*4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(planes*4)

        self.global_avg = nn.AvgPool2d(ave_kernel)
        self.down = nn.Conv2d(planes * 4, int(planes / 4), kernel_size=1, stride=1, padding=0, bias=True)
        self.up = nn.Conv2d(int(planes / 4), planes * 4, kernel_size=1, stride=1, padding=0, bias=True)
        # self.fc1 = nn.Linear(planes * 4, int(planes / 4))
        # self.fc2 = nn.Linear(int(planes / 4), planes * 4)
        
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        branch1 = self.conv1_1(x)
        branch1 = self.bn1_1(branch1)
        branch1 = self.relu(branch1)
        branch1 = self.conv1_2(branch1)

        branch2 = self.conv2_1(x)
        branch2 = self.bn2_1(branch2)
        branch2 = self.relu(branch2)
        branch2 = self.conv2_2(branch2)
        branch2 = self.bn2_2(branch2)
        branch2 = self.relu(branch2)
        branch2 = self.conv2_3(branch2)

        out = torch.cat((branch1, branch2), 1)
        out = self.bn_concat(out)
        out = self.relu(out)
  
        out = self.conv(out)
        out = self.bn(out)

        se = self.global_avg(out)
        # se = se.view(se.size(0), -1)
        se = self.down(se)
        se = self.relu(se)
        se = self.up(se)
        se = self.sigmoid(se)
        # se = se.view(se.size(0), se.size(1), 1, 1)

        out = out * se.expand_as(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SE_AIR(nn.Module):
    def __init__(self, layers, num_classes):
        """ Constructor
        Args:
            layers: config of layers, e.g., [3, 4, 6, 3]
            num_classes: number of classes
        """
        super(SE_AIR, self).__init__()
        block = SEBottleneck

        self.num_classes = num_classes
        self.inplanes = 128
        
        # self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)
        
        self.conv1 = nn.Conv2d(3, 64, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], ak=56)
        self.layer2 = self._make_layer(block, 128, layers[1], 2, ak=28)
        self.layer3 = self._make_layer(block, 256, layers[2], 2, ak=14)
        self.layer4 = self._make_layer(block, 512, layers[3], 2, ak=7)
        self.avgpool = nn.AvgPool2d(7)      
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, ak=56):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.
        Args:
            block: block type used to construct ResNet
            planes: number of output channels (need to multiply by block.expansion)
            blocks: number of blocks to be built
            stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.
        Returns: a Module consisting of n sequential bottlenecks.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, ave_kernel=ak))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, ave_kernel=ak))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def se_air26():
    """
    Construct SE-AIR-26.
    """
    model = SE_AIR([2, 2, 2, 2], 1000)
    return model

def se_air50():
    """
    Construct SE-AIR-50.
    """
    model = SE_AIR([3, 4, 6, 3], 1000)
    return model

def se_air101():
    """
    Construct SE-AIR-101, 200 layers.
    """
    model = SE_AIR([3, 4, 23, 3], 1000)
    return model

def se_air152():
    """
    Construct SE-AIR-152, 302 layers.
    """
    model = SE_AIR([3, 8, 36, 3], 1000)
    return model
