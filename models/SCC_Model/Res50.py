import torch.nn as nn
import torch
from torchvision import models

from misc.layer import Conv2d, FC

import torch.nn.functional as F
from misc.utils import *

import pdb

model_path = '../PyTorch_Pretrained/resnet50-19c8e357.pth'


class Res50(nn.Module):
    def __init__(self, pretrained=True):
        super(Res50, self).__init__()

        self.de_pred = nn.Sequential(Conv2d(128, 64, 1, same_padding=True, NL='relu'),
                                     Conv2d(64, 1, 1, same_padding=True, NL='relu'))

        # initialize_weights(self.modules())

        self.res = models.resnet50(pretrained=pretrained)
        # pre_wts = torch.load(model_path)
        # res.load_state_dict(pre_wts)
        #         self.frontend = nn.Sequential(
        #             res.conv1, res.bn1, res.relu, res.maxpool, res.layer1, res.layer2
        #         )
        self.own_reslayer_3 = make_res_layer(Bottleneck, 256, 6, stride=1)
        self.own_reslayer_3.load_state_dict(self.res.layer3.state_dict())
        self.own = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=3, stride=2, bias=False,padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
        )
        # 修改下采样最后一层
        downsample = nn.Sequential(
            nn.Conv2d(1024, 2048,
                      kernel_size=3, stride=2, bias=False,padding=1),
            nn.BatchNorm2d(2048),
        )
        self.res4=Bottleneck(1024, 512, stride=2, downsample=downsample)
        self.latter1 = nn.Conv2d(2048, 128, kernel_size=1, stride=1, padding=0)
        self.latter2 = nn.Conv2d(1024, 128, kernel_size=1, stride=1, padding=0)
        self.latter3 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.latter4 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.latter5 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)
        self.wh_layer = nn.Conv2d(128, 2, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, x):

        x = self.res.conv1(x)
        x = self.res.bn1(x)
        x1 = self.res.relu(x)
        x2 = self.res.maxpool(x1)
        x3 = self.res.layer1(x2)
        x4 = self.res.layer2(x3)  # [bs,128,72,96]

        x5 = self.own_reslayer_3(x4)

        # x6 = self.own(x5)
        x6=self.res4(x5)
        p6 = self.latter1(x6)
        p5 = self._upsample_add(p6, self.latter2(x5))
        p4 = p5 + self.latter3(x4)
        p3 = self._upsample_add(p4, self.latter4(x3))
        p2 = p3 + self.latter5(x2)  # [bs,128,144,192]

        hm = self.de_pred(p2)
        hm = F.upsample(hm, scale_factor=4)
        wh = self.wh_layer(p2)
        wh = F.upsample(wh, scale_factor=4)  # [bs,2,128,128]
        #         print(wh.shape)

        return hm, wh

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, std=0.01)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.fill_(1)
                m.bias.data.fill_(0)


def make_res_layer(block, planes, blocks, stride=1):
    downsample = None
    inplanes = 512
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(block(inplanes, planes, stride, downsample))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes, planes))

    return nn.Sequential(*layers)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
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