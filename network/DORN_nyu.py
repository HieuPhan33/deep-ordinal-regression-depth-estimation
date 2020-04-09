# -*- coding: utf-8 -*-
# @Time    : 2018/11/22 12:33
# @Author  : Wang Xin
# @Email   : wangxin_buaa@163.com
import os

import torch
import torch.nn as nn
import math
from torchvision import models
#from modules import *
from .modules import *
import config

from network.backbone import resnet101,resnet18

class FullImageEncoder(nn.Module):
    def __init__(self,channels,input_size,output_size):
        super(FullImageEncoder, self).__init__()
        self.channels = channels
        self.input_size = input_size
        self.r = 2
        self.global_pooling = nn.AvgPool2d(kernel_size=self.r
                                           , stride=self.r)  # KITTI 16 16
        self.dropout = nn.Dropout2d(p=0.5)
        self.global_fc = nn.Linear(self.channels*(input_size[0]//self.r)* (input_size[1]//self.r),
                                   512)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(512, 512, 1)  # Cross-channel projection
        self.upsample = nn.UpsamplingBilinear2d(size=output_size)  # Copy to Feature vector of NYU 33X45

        weights_init(self.modules(), 'xavier')

    def forward(self, x):
        x1 = self.global_pooling(x)
        # print('# x1 size:', x1.size())
        x2 = self.dropout(x1)
        x3 = x2.view(-1, self.channels * (self.input_size[0]//self.r) * (self.input_size[1]//self.r)) # Flatten out ready to leaarn globally by FC
        x4 = self.relu(self.global_fc(x3))
        # print('# x4 size:', x4.size())
        x4 = x4.view(-1, 512, 1, 1)
        # print('# x4 size:', x4.size())
        x5 = self.conv1(x4)
        out = self.upsample(x5) # Copy
        return out



class SceneUnderstandingModule(nn.Module):
    def __init__(self,output_size,input_size,total_label):
        super(SceneUnderstandingModule, self).__init__()
        total_ord_label = total_label # For NYU
        self.input_size= input_size
        #upsampling_size = (33,45)
        upsampling_size = tuple(int(i*2) for i in input_size)
        #total_ord_label = 90 # For uow
        self.channels = 2048
        self.encoder = FullImageEncoder(self.channels,self.input_size,
                                        output_size=upsampling_size)
        #total_K = (total_ord_label-1)*2 For probabilistic inference
        total_K = total_ord_label*2
        self.aspp1 = nn.Sequential(
            nn.Conv2d(self.channels, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=upsampling_size)
        )
        self.aspp2 = nn.Sequential(
            nn.Conv2d(self.channels, 512, 3, padding=6, dilation=6), # w + 1
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=upsampling_size)
        )
        self.aspp3 = nn.Sequential(
            nn.Conv2d(self.channels, 512, 3, padding=12, dilation=12),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=upsampling_size)
        )
        self.aspp4 = nn.Sequential(
            nn.Conv2d(self.channels, 512, 3, padding=18, dilation=18),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=upsampling_size)
        )
        # self.concat_process = nn.Sequential(
        #     nn.Dropout2d(p=0.5),
        #     nn.Conv2d(512 * 5, self.channels, 1),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout2d(p=0.5),
        #     nn.Conv2d(self.channels, total_K, 1),  # Number of labels : KITTI 71 NYU 68
        #     # nn.UpsamplingBilinear2d(scale_factor=8)
        #     nn.UpsamplingBilinear2d(size=(257, 353)),
        #     nn.Conv2d(total_K,total_K, 1)
        # )
        r = 2
        self.concat_process = nn.Sequential(
            nn.Dropout2d(p=0.5), 
            #ChannelSELayer(num_channels=512*5,reduction_ratio=r),
            #ChannelSpatialSELayer(num_channels=512*5,reduction_ratio=r),
            #SpatialSELayer(num_channels=512*5),
            ChannelwiseLocalAttention(pooling_output_size=(upsampling_size[0]//r,
                                                           upsampling_size[1]//r),n_heads=2),
            #AugmentedConv(in_channels=512*5,out_channels=512*5,kernel_size=3, dk=40, dv=4, Nh=2, relative=False, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512*5,self.channels,1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(self.channels,total_K,1),
            #nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=output_size),
            #nn.Conv2d(total_K, total_K,1),
            #nn.ReLU(inplace=True)
        )

        weights_init(self.modules(), type='xavier')

    def forward(self, x):
        x1 = self.encoder(x)

        x2 = self.aspp1(x)
        x3 = self.aspp2(x)
        x4 = self.aspp3(x)
        x5 = self.aspp4(x)

        x6 = torch.cat((x1, x2, x3, x4, x5), dim=1)
        # print('cat x6 size:', x6.size())
        out = self.concat_process(x6)
        return out


class DORN(nn.Module):
    def __init__(self, output_size=(257, 353),total_label=config.make3d_K, channel=3, pretrained=True, freeze=True):
        super(DORN, self).__init__()

        self.output_size = output_size
        self.channel = channel
        self.encoder_output_size = tuple(i//32 for i in output_size)
        #self.feature_extractor = resnet101(pretrained=pretrained)
        #self.feature_extractor = models.resnet18(pretrained=pretrained)
        #self.feature_extractor = resnet18(pretrained=pretrained)
        model = models.resnet50(pretrained=True)
        #model = models.resnet50(pretrained=pretrained)
        self.feature_extractor = torch.nn.Sequential(*(list(model.children())[:-2]))
        self.aspp_module = SceneUnderstandingModule(output_size,self.encoder_output_size,total_label)
        self.orl = OrdinalRegressionLayer()

    def forward(self, x):
        x1 = self.feature_extractor(x)
        # print(x1.size())
        x2 = self.aspp_module(x1)
        # print('DORN x2 size:', x2.size())
        depth_labels, ord_labels = self.orl(x2)
        return depth_labels, ord_labels

    def get_1x_lr_params(self):
        b = [self.feature_extractor]
        for i in range(len(b)):
            for k in b[i].parameters():
                if k.requires_grad:
                    yield k

    def get_10x_lr_params(self):
        b = [self.aspp_module, self.orl]
        for j in range(len(b)):
            for k in b[j].parameters():
                if k.requires_grad:
                    yield k


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 默认使用GPU 0

if __name__ == "__main__":
    model = DORN(pretrained=False,output_size=config.kitti_output_size)
    model = model.cuda()
    model.eval()
    s = config.kitti_output_size
    image = torch.randn(1, 3, s[0], s[1])
    image = image.cuda()
    with torch.no_grad():
        out0, out1 = model(image)
    print('out0 size:', out0.size())
    print('out1 size:', out1.size())

    print(out0)
