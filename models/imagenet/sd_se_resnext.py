from __future__ import division

""" 
Creates a SE-ResNeXt with Stochastic Depth Model as defined in:
Xie, S., Girshick, R., Dollar, P., Tu, Z., & He, K. (2016). 
Aggregated residual transformations for deep neural networks. 
"""
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch

__all__ = ['sdse_resnext26', 'sdse_resnext50', 'sdse_resnext101', 'sdse_resnext152']


class SDSEBottleneck(nn.Module):
    """
    SDSERexNeXt bottleneck type C
    """
    expansion = 4

    def __init__(self, inplanes, planes, baseWidth, cardinality, stride=1, downsample=None, death_rate=0., ave_kernel=56):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            baseWidth: base width.
            cardinality: num of convolution groups.
            stride: conv stride. Replaces pooling layer.
        """
        super(SDSEBottleneck, self).__init__()

        D = int(math.floor(planes * (baseWidth / 64)))
        C = cardinality

        self.conv1 = nn.Conv2d(inplanes, D * C, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(D * C)
        self.conv2 = nn.Conv2d(D * C, D * C, kernel_size=3, stride=stride, padding=1, groups=C, bias=False)
        self.bn2 = nn.BatchNorm2d(D * C)
        self.conv3 = nn.Conv2d(D * C, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.global_avg = nn.AvgPool2d(ave_kernel)
        self.fc1 = nn.Linear(planes * 4, int(planes / 4))
        self.fc2 = nn.Linear(int(planes / 4), planes * 4)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.death_rate = death_rate
        self.downsample = downsample

    def forward(self, x):
        residual = x

        if self.downsample is not None:
            x = self.downsample(x)

        if not self.training or torch.rand(1)[0] >= self.death_rate:
            residual = self.conv1(residual)
            residual = self.bn1(residual)
            residual = self.relu(residual)

            residual = self.conv2(residual)
            residual = self.bn2(residual)
            residual = self.relu(residual)

            residual = self.conv3(residual)
            residual = self.bn3(residual)

            se = self.global_avg(residual)
            se = se.view(se.size(0), -1)
            se = self.fc1(se)
            se = self.relu(se)
            se = self.fc2(se)
            se = self.sigmoid(se)
            se = se.view(se.size(0), se.size(1), 1, 1)

            residual = residual * se.expand_as(residual)

            if self.training:
                residual /= (1. - self.death_rate)

            x = x + residual
            x = self.relu(x)

        return x


class SDSE_ResNeXt(nn.Module):
    """
    SDSERexNeXt optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1611.05431.pdf
    """

    def __init__(self, baseWidth, cardinality, layers, num_classes, death_rates=None):
        """ Constructor
        Args:
            baseWidth: baseWidth for ResNeXt.
            cardinality: number of convolution groups.
            layers: config of layers, e.g., [3, 4, 6, 3]
            num_classes: number of classes
        """
        super(SDSE_ResNeXt, self).__init__()
        block = SDSEBottleneck

        assert death_rates is None or len(death_rates) == sum(layers)
        if death_rates is None:
            death_rates = [0.] * sum(layers)

        self.cardinality = cardinality
        self.baseWidth = baseWidth
        self.num_classes = num_classes
        self.inplanes = 64
        self.output_size = 64

        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], death_rates[:sum(layers[:1])], ak=56)
        self.layer2 = self._make_layer(block, 128, layers[1], death_rates[sum(layers[:1]):sum(layers[:2])], 2, ak=28)
        self.layer3 = self._make_layer(block, 256, layers[2], death_rates[sum(layers[:2]):sum(layers[:3])], 2, ak=14)
        self.layer4 = self._make_layer(block, 512, layers[3], death_rates[sum(layers[:3]):sum(layers)], 2, ak=7)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, death_rates, stride=1, ak=56):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.
        Args:
            block: block type used to construct ResNext
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
        layers.append(block(self.inplanes, planes, self.baseWidth, self.cardinality, stride, downsample, death_rate=death_rates[0], ave_kernel=ak))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.baseWidth, self.cardinality, death_rate=death_rates[i], ave_kernel=ak))

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


def generate_death_rates(death_mode='none', death_rate=0.5, nblocks=16):
    if death_mode == 'uniform':
        death_rates = [death_rate] * nblocks
    elif death_mode == 'linear':
        death_rates = [float(i + 1) * death_rate / float(nblocks)
                       for i in range(nblocks)]
    else:
        death_rates = None

    return death_rates


def sdse_resnext26(baseWidth=4, cardinality=32):
    """
    Construct SDSE-ResNeXt-26. 
    """
    death_rates = generate_death_rates(death_mode='linear', death_rate=0.5, nblocks=8)
    print death_rates
    model = SDSE_ResNeXt(baseWidth, cardinality, [2, 2, 2, 2], 1000, death_rates)
    return model


def sdse_resnext50(baseWidth=4, cardinality=32):
    """
    Construct SDSE-ResNeXt-50.
    """
    death_rates = generate_death_rates(death_mode='linear', death_rate=0.5, nblocks=16)
    print death_rates
    model = SDSE_ResNeXt(baseWidth, cardinality, [3, 4, 6, 3], 1000, death_rates)
    return model


def sdse_resnext101(baseWidth=4, cardinality=32):
    """
    Construct SDSE-ResNeXt-101.
    """
    death_rates = generate_death_rates(death_mode='linear', death_rate=0.5, nblocks=33)
    print death_rates
    model = SDSE_ResNeXt(cardinality, baseWidth, [3, 4, 23, 3], 1000, death_rates)
    return model


def sdse_resnext152(baseWidth=4, cardinality=32):
    """
    Construct SDSE-ResNeXt-152.
    """
    death_rates = generate_death_rates(death_mode='linear', death_rate=0.5, nblocks=50)
    print death_rates
    model = SDSE_ResNeXt(baseWidth, cardinality, [3, 8, 36, 3], 1000, death_rates)
    return model
