# -*- coding: utf-8 -*-
# @Time    : 2018/11/22 12:33
# @Author  : Wang Xin
# @Email   : wangxin_buaa@163.com
import os

import torch
import torch.nn as nn
import math
from torchvision import models

from network.backbone import resnet101,resnet18


def weights_init(modules, type='xavier'):
    m = modules
    if isinstance(m, nn.Conv2d):
        if type == 'xavier':
            torch.nn.init.xavier_normal_(m.weight)
        elif type == 'kaiming':  # msra
            torch.nn.init.kaiming_normal_(m.weight)
        else:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))

        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        if type == 'xavier':
            torch.nn.init.xavier_normal_(m.weight)
        elif type == 'kaiming':  # msra
            torch.nn.init.kaiming_normal_(m.weight)
        else:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))

        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        if type == 'xavier':
            torch.nn.init.xavier_normal_(m.weight)
        elif type == 'kaiming':  # msra
            torch.nn.init.kaiming_normal_(m.weight)
        else:
            m.weight.data.fill_(1.0)

        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Module):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                if type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight)
                elif type == 'kaiming':  # msra
                    torch.nn.init.kaiming_normal_(m.weight)
                else:
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))

                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                if type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight)
                elif type == 'kaiming':  # msra
                    torch.nn.init.kaiming_normal_(m.weight)
                else:
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))

                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight)
                elif type == 'kaiming':  # msra
                    torch.nn.init.kaiming_normal_(m.weight)
                else:
                    m.weight.data.fill_(1.0)

                if m.bias is not None:
                    m.bias.data.zero_()


class FullImageEncoder(nn.Module):
    def __init__(self,channels):
        super(FullImageEncoder, self).__init__()
        self.channels = channels
        self.global_pooling = nn.AvgPool2d(kernel_size=3
                                           , stride=3)  # KITTI 16 16
        self.dropout = nn.Dropout2d(p=0.5)
        self.global_fc = nn.Linear(self.channels*3*4, 512)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(512, 512, 1)  # Cross-channel projection
        self.upsample = nn.UpsamplingBilinear2d(size=(33, 45))  # Copy to Feature vector of NYU 33X45

        weights_init(self.modules(), 'xavier')

    def forward(self, x):
        x1 = self.global_pooling(x)
        # print('# x1 size:', x1.size())
        x2 = self.dropout(x1)
        x3 = x2.view(-1, self.channels * 3 * 4) # Flatten out ready to leaarn globally by FC
        x4 = self.relu(self.global_fc(x3))
        # print('# x4 size:', x4.size())
        x4 = x4.view(-1, 512, 1, 1)
        # print('# x4 size:', x4.size())
        x5 = self.conv1(x4)
        out = self.upsample(x5) # Copy
        return out

class SceneUnderstandingModule(nn.Module):
    def __init__(self):
        super(SceneUnderstandingModule, self).__init__()
        total_ord_label = 68 # For NYU
        self.channels = 512
        self.encoder = FullImageEncoder(self.channels)
        total_K = (total_ord_label-1)*2
        self.aspp1 = nn.Sequential(
            nn.Conv2d(self.channels, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(33, 45))
        )
        self.aspp2 = nn.Sequential(
            nn.Conv2d(self.channels, 512, 3, padding=6, dilation=6), # w + 1
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(33, 45))
        )
        self.aspp3 = nn.Sequential(
            nn.Conv2d(self.channels, 512, 3, padding=12, dilation=12),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(33, 45))
        )
        self.aspp4 = nn.Sequential(
            nn.Conv2d(self.channels, 512, 3, padding=18, dilation=18),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(33, 45))
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

        weights_init(self.modules(), type='xavier')

    def forward(self, x):
        x1 = self.encoder(x)

        x2 = self.aspp1(x)
        x3 = self.aspp2(x)
        x4 = self.aspp3(x)
        x5 = self.aspp4(x)

        x6 = torch.cat((x1, x2, x3, x4, x5), dim=1)
        # print('cat x6 size:', x6.size())
        #out = self.concat_process(x6)
        out = x6
        return out


def upconv(in_channels, out_channels, kernel_size=5, output_size=None):
    # Unpool then conv maintaining resolution

    modules = [
        nn.UpsamplingBilinear2d(scale_factor=2),
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=2, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    ]
    if output_size:
        modules.append(nn.UpsamplingNearest2d(size=output_size))
    return nn.Sequential(*modules)

class OrdinalRegressionLayer(nn.Module):
    def __init__(self):
        super(OrdinalRegressionLayer, self).__init__()

    def forward(self, x):
        """
        :param x: N x H x W x C, N is batch_size, C is channels of features
        :return: ord_labels is ordinal outputs for each spatial locations , size is N x H X W X C (C = 2K, K is interval of SID)
                 decode_label is the ordinal labels for each position of Image I
        """
        N, C, H, W = x.size()
        ord_num = C // 2

        """
        replace iter with matrix operation
        fast speed methods
        """
        A = x[:, ::2, :, :].clone() # Took every odd-th element
        B = x[:, 1::2, :, :].clone() # Took every even-th element

        A = A.view(N, 1, ord_num * H * W) # Trick to combine, expand all pixels by pixels
        B = B.view(N, 1, ord_num * H * W)

        C = torch.cat((A, B), dim=1) # Soft-max each binary response
        C = torch.clamp(C, min=1e-8, max=1e8)  # prevent nans

        ord_c = nn.functional.softmax(C, dim=1)

        ord_c1 = ord_c[:, 1, :].clone() # Response corresponding to 1
        ord_c1 = ord_c1.view(-1, ord_num, H, W)
        # print('ord > 0.5 size:', (ord_c1 > 0.5).size())

        # Original summing for rank
        decode_c = torch.sum((ord_c1 > 0.5), dim=1).view(-1, 1, H, W) # The one-label pixel corresponds to one rank

        # Create new temp matrix to compute probability corresponding to each rank r:
        # Pr(k=r) = Pr(k>r-1) - Pr(k>r) and Pr(k=0) = 1-Pr(k>0) and Pr(k=K) = Pr(k>K-1)
        # temp = torch.zeros((N, ord_num+ 2, H, W))
        # temp[:, 0, :, :] = torch.ones((N, H, W))
        # temp[:, 1:ord_num + 1, :, :] = ord_c1
        # prob = torch.zeros((N,ord_num+1,H,W))
        # for i in range(ord_num+1):
        #     prob[:,i,:,:] = temp[:,i+1,:,:] - temp[:,i,:,:]
        # decode_c = torch.argmax(prob,dim=1).view(-1,1,H,W) # Matching the shape of the target -> N x 1 x H x W
        # # Derive rank based on probabilistic
        # if torch.cuda.is_available():
        #     decode_c = decode_c.cuda()

        return decode_c, ord_c1


class DORN(nn.Module):
    def __init__(self, output_size=(257, 353), channel=3, total_ord_label=68, pretrained=True, freeze=True):
        super(DORN, self).__init__()

        self.output_size = output_size
        self.channel = channel
        #self.feature_extractor = resnet101(pretrained=pretrained)
        #self.feature_extractor = models.resnet18(pretrained=pretrained)
        #self.feature_extractor = resnet18(pretrained=pretrained)
        model = models.resnet18(pretrained=True)
        self.feature_extractor = torch.nn.Sequential(*(list(model.children())[:-2]))
        self.aspp_module = SceneUnderstandingModule()


        # Multi-task predictor
        #total_K = (total_ord_label-1)*2
        total_K = total_ord_label*2 # For original summing
        self.ordinal_regressor = nn.Sequential(
            nn.Dropout2d(p=0.5),
            nn.Conv2d(512 * 5, 512, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(512, total_K, 1),  # Number of labels : KITTI 71 NYU 68
            # nn.UpsamplingBilinear2d(scale_factor=8)
            nn.UpsamplingBilinear2d(size=(257, 353)),
            nn.Conv2d(total_K,total_K, 1),
            nn.ReLU()
        )
        self.regressor = nn.Sequential(
            nn.Dropout2d(p=0.5),
            nn.Conv2d(512 * 5, 512, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            upconv(512,256),
            nn.Dropout2d(p=0.5),
            upconv(256,128),
            nn.Dropout2d(p=0.5),
            upconv(128,64),
            nn.Dropout2d(p=0.5),
            upconv(64,32),
            nn.Dropout2d(p=0.5),
            nn.UpsamplingBilinear2d(size=(257, 353)),
            nn.Conv2d(32,1, 1),
            nn.ReLU(inplace=True)
        )
        self.orl = OrdinalRegressionLayer()

    def forward(self, x):
        x1 = self.feature_extractor(x)
        # print(x1.size())
        x2 = self.aspp_module(x1)
        or_output = self.ordinal_regressor(x2)
        r_output = self.regressor(x2)
        # print('DORN x2 size:', x2.size())
        depth_labels, ord_labels = self.orl(or_output)

        return depth_labels, ord_labels, r_output

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
    model = DORN(pretrained=False)
    model = model.cuda()
    model.eval()
    image = torch.randn(1, 3, 257, 353)
    image = image.cuda()
    with torch.no_grad():
        out0, out1 = model(image)
    print('out0 size:', out0.size())
    print('out1 size:', out1.size())

    print(out0)
