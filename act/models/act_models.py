import torch.nn as nn
import math
from .act_factory import *

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, act = 'ReLU', downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        if act == 'ReLU':
            self.act1 = nn.ReLU()
            self.act2 = nn.ReLU()
        elif act == 'Mish':
            self.act1 = Mish()
            self.act2 = Mish()
        elif act == 'Swish':
            self.act1 = Swish()
            self.act2 = Swish()
        elif act == 'Funnel':
            self.act1 = FReLU(planes)
            self.act2 = FReLU(planes)
        elif act == 'DYReLUA':
            self.act1 = DyReLUA(planes)
            self.act2 = DyReLUA(planes)
        else:
            self.act1 = DyReLUB(planes)
            self.act2 = DyReLUB(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.act2(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, act='ReLU', downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        if act == 'ReLU':
            self.act1 = nn.ReLU()
            self.act2 = nn.ReLU()
            self.act3 = nn.ReLU()
        elif act == 'Mish':
            self.act1 = Mish()
            self.act2 = Mish()
            self.act3 = Mish()
        elif act == 'Swish':
            self.act1 = Swish()
            self.act2 = Swish()
            self.act3 = Swish()
        elif act == 'Funnel':
            self.act1 = FReLU(planes)
            self.act2 = FReLU(planes)
            self.act2 = FReLU(planes*4)
        elif act == 'DYReLUA':
            self.act1 = DyReLUA(planes)
            self.act2 = DyReLUA(planes)
            self.act2 = DyReLUA(planes*4)
        else:
            self.act1 = DyReLUB(planes)
            self.act2 = DyReLUB(planes)
            self.act2 = DyReLUB(planes*4)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.act3(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, act, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        if act == 'ReLU':
            self.act = nn.ReLU()
        elif act == 'Mish':
            self.act = Mish()
        elif act == 'Swish':
            self.act = Swish()
        elif act == 'Funnel':
            self.act = FReLU(64)
        elif act == 'DYReLUA':
            self.act = DyReLUA(64)
        else:
            self.act = DyReLUB(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, act, 64, layers[0])
        self.layer2 = self._make_layer(block, act, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, act, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, act, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, act, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, act, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, act=act))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(act = "ReLU", num_classes=1_000, pretrained=False):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_classes:The classes of classification
    """
    model = ResNet(BasicBlock, act, [2, 2, 2, 2], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def resnet34(act = "ReLU", num_classes=1_000, pretrained=False):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_classes:The classes of classification
    """
    model = ResNet(BasicBlock, act, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def resnet50(act = "ReLU", num_classes=1000, pretrained=False):
    """Constructs a ResNet-50 model.

    Args:
        num_classes:The classes of classification
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, act, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model
