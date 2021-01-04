import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


## ResNet-50 definition
class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1      = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1        = nn.BatchNorm2d(planes)
        self.conv2      = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3*dilation-1)//2, bias=False, dilation=dilation)
        self.bn2        = nn.BatchNorm2d(planes)
        self.conv3      = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3        = nn.BatchNorm2d(planes*4)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            x = self.downsample(x)
        return F.relu(out+x, inplace=True)

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1    = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1      = nn.BatchNorm2d(64)
        self.layer1   = self.make_layer( 64, 3, stride=1, dilation=1)
        self.layer2   = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3   = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4   = self.make_layer(512, 3, stride=2, dilation=1)

        self.initialize()

    def make_layer(self, planes, blocks, stride, dilation):
        downsample    = nn.Sequential(nn.Conv2d(self.inplanes, planes*4, kernel_size=1, stride=stride, bias=False),
                                      nn.BatchNorm2d(planes*4))
        layers        = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes*4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out1 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out2, out3, out4, out5

    def initialize(self):
        logging.info(f'Loading ResNet-50 pretrained model')
        self.load_state_dict(torch.load('./pretrained/resnet50-19c8e357.pth'), strict=False)
        logging.info(f'Loading ResNet-50 pretrained model is done!')


## TODO - Define a simple network (decoder part), which will use ResNet-50 as an encoder.

class BottleneckReverse(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BottleneckReverse, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3 * dilation - 1) // 2,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes // 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes // 4)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            x = self.downsample(x)

        return F.relu(out + x, inplace=True)


class ResNetReverse(nn.Module):
    def __init__(self):
        super(ResNetReverse, self).__init__()
        # self.inplanes = 64
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.layer1 = self.make_layer(64, 3, stride=1, dilation=1)
        # self.layer2 = self.make_layer(128, 4, stride=2, dilation=1)
        # self.layer3 = self.make_layer(256, 6, stride=2, dilation=1)
        # self.layer4 = self.make_layer(512, 3, stride=2, dilation=1)

        self.inplanes = 512
        self.upconv1_1 = nn.ConvTranspose2d(2048, 512, kernel_size=2, stride=2, dilation=4, bias=False)
        self.upconv1_2 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2, dilation=4, bias=False)
        self.layer1 = self.make_layer(512, 3, stride=1, dilation=1)

        self.upconv2_1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, dilation=4, bias=False)
        self.upconv2_2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, dilation=4, bias=False)
        self.layer2 = self.make_layer(256, 4, stride=2, dilation=1)

        self.upconv3_1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, dilation=4, bias=False)
        self.upconv3_2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, dilation=4, bias=False)
        self.layer3 = self.make_layer(128, 6, stride=2, dilation=1)

        self.upconv4_1 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, dilation=4, bias=False)
        self.upconv4_2 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=1, dilation=1, bias=False)
        self.layer4 = self.make_layer(64, 3, stride=2, dilation=1)

        self.upconv5 = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2, dilation=1, bias=False)
        self.conv1 = nn.Conv2d(16, 1, kernel_size=5, stride=1, padding=0, bias=False, dilation=34)


    def make_layer(self, planes, blocks, stride, dilation):
        downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes // 4, kernel_size=1, stride=stride, bias=False),
                                   nn.BatchNorm2d(planes // 4))
        # print(self.inplanes, planes)
        layers = [BottleneckReverse(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes // 4
        for _ in range(1, blocks):
            layers.append(BottleneckReverse(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.upconv1_1(x)
        out1 = self.upconv1_2(out1)
        # print("out1 upconv shape: ", out1.shape)
        out1 = self.layer1(out1)
        # print("out1.shape: ", out1.shape)

        out2 = self.upconv2_1(out1)
        out2 = self.upconv2_2(out2)
        # print("out2 upconv shape: ", out2.shape)
        out2 = self.layer2(out2)
        # print("out2.shape: ", out2.shape)

        out3 = self.upconv3_1(out2)
        out3 = self.upconv3_2(out3)
        # print("out3 upconv shape: ", out3.shape)
        out3 = self.layer3(out3)
        # print("out3.shape: ", out3.shape)

        out4 = self.upconv4_1(out3)
        out4 = self.upconv4_2(out4)
        # print("out4 upconv shape: ", out4.shape)
        out4 = self.layer4(out4)
        # print("out4.shape: ", out4.shape)

        out5 = self.upconv5(out4)
        # print("out5 upconv shape: ", out5.shape)
        out5 = self.conv1(out5)
        # print("out5.shape: ", out5.shape)

        return out2, out3, out4, out5


class SimpleNet(nn.Module):
    def __init__(self, n_classes=1):
        super(SimpleNet, self).__init__()
        self.bkbone = ResNet()
        self.decoder = ResNetReverse()

    def forward(self, x):
        out2, out3, out4, out5 = self.bkbone(x)

        logits = self.decoder(out5)
        return logits
#
#
# class Decoder(nn.Module):
#     def __init__(self):
#         super(Decoder, self).__init__()
#
#     def forward(self, x):
#         pass
#
#
#
#
#
# class OutConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(OutConv, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#
#     def forward(self, x):
#         return self.conv(x)
#
# class SimpleNet(nn.Module):
#     def __init__(self, n_classes=1):
#         super(SimpleNet, self).__init__()
#         self.n_classes = n_classes
#         self.bkbone = ResNet()
#
#         ...
#
#         ## previous_layer_channels is the number of penultimate layer's channels.
#         self.outc = OutConv(previous_layer_channels, n_classes)
#
#     def forward(self, x):
#         out2, out3, out4, out5 = self.bkbone(x)
#
#         ...
#
#         logits = self.outc(previous_layer)
#         return logits
#