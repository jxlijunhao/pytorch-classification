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

__all__ = ['air50', 'air101', 'air152']


class AttentionBlock_A(nn.Module):
    """
    AttentionBlock_A bottleneck
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
        """
        super(AttentionBlock_A, self).__init__()
        
        self.downsample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        pre_block = AIRBottleneck(inplanes, planes)
        down2_block1 = AIRBottleneck(inplanes, planes)
        down2_block2 = AIRBottleneck(inplanes, planes)
        down4_block1 = AIRBottleneck(inplanes, planes)
        down4_block2 = AIRBottleneck(inplanes, planes) 
        down8_block1 = AIRBottleneck(inplanes, planes)
        down8_block2 = AIRBottleneck(inplanes, planes)        
       
        up2_block1 = AIRBottleneck(inplanes, planes)
        up4_block1 = AIRBottleneck(inplanes, planes)
        
        self.mask_conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.maks_bn1 = nn.BatchNorm2d(inplanes)
        self.mask_conv2 = nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.maks_bn2 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        block1 = AIRBottleneck(inplanes, planes)
        block2 = AIRBottleneck(inplanes, planes)    
        pos_block = AIRBottleneck(inplanes, planes)
           
    def forward(self, x):
        x = self.pre_block(x)
        
        d2 = self.downsample(x)
        d2 = self.down2_block1(d2)
        d22 = self.down2_block2(d2)

        d4 = self.downsample(d2)
        d4 = self.down4_block1(d4)
        d42 = self.down4_block2(d4)
            
        d8 = self.downsample(d4)
        d8 = self.down8_block1(d8)
        d82 = self.down8_block2(d8)
        
        up2 = d42 + F.upsample(d82, scale_factor=2)
        up2 = self.relu(up2)
        up2 = self.up2_block1(up2)
        
        up4 = d22 + F.upsample(up2, scale_factor=2)
        up4 = self.relu(up4)
        up4 = self.up4_block1(up4)
        
        up8 = F.upsample(up4, scale_factor=2)
        up8 = self.mask_conv1(up8)
        up8 = self.mask_bn1(up8)
        up8 = self.relu(up8)
        up8 = self.mask_conv2(up8)
        up8 = self.mask_bn2(up8)
        mask = self.sigmoid(up8)   
        
        x = self.block1(x)
        x = self.block2(x)
        attention = x * mask
        x += attention
        x = self.relu(x)
        out = self.pos_block(x)
        
        return out
        

class AIRBottleneck(nn.Module):
    """
    AIRBottleneck bottleneck
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
        """
        super(AIRBottleneck, self).__init__()	

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

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class AIR(nn.Module):
    def __init__(self, layers, num_classes):
        """ Constructor
        Args:
            layers: config of layers, e.g., [3, 4, 6, 3]
            num_classes: number of classes
        """
        super(AIR, self).__init__()
        block = AIRBottleneck

        self.num_classes = num_classes
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], 2)
        self.layer3 = self._make_layer(block, 256, layers[2], 2)
        self.layer4 = self._make_layer(block, 512, layers[3], 2)
        self.avgpool = nn.AvgPool2d(7)      
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
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
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def air50():
    """
    Construct AIR-50.
    """
    model = AIR([3, 4, 6, 3], 1000)
    return model

def air101():
    """
    Construct AIR-101.
    """
    model = AIR([3, 4, 23, 3], 1000)
    return model

def air152():
    """
    Construct AIR-152.
    """
    model = AIR([3, 8, 36, 3], 1000)
    return model
