# -*- coding: utf-8 -*-
# @Time    : 2018/11/22 12:33
# @Author  : Wang Xin
# @Email   : wangxin_buaa@163.com
import os

import torch
import torch.nn as nn
import math
from torchvision import models
import torch.nn.functional as F

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

class ChannelwiseLocalAttention(nn.Module):
    def __init__(self, h_size = 0, pooling_output_size=(4, 4),n_heads=1):
        super(ChannelwiseLocalAttention, self).__init__()
        self.pooling_output_size = pooling_output_size
        self.n_heads = n_heads
        # self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=kernel_size)
        #self.pool = nn.AdaptiveAvgPool2d(output_size=pooling_output_size)
        in_channels = pooling_output_size[0] * pooling_output_size[1]
        if h_size == 0:
            h_size = in_channels
        if h_size == 0:
            self.h_size = in_channels
        self.h_size = h_size
        assert(in_channels % n_heads == 0, "n_heads must be divisible by in_channels")
        out_channels = self.h_size * self.n_heads
        # Each conv_matrix having shape of 1 x 1 x (H*W) x (H*W/r)
        # They will be convolved on channel-wise matrix of shape (H*W) * C
        self.conv_Q = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, groups=n_heads, kernel_size=1)
        self.conv_K = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, groups=n_heads, kernel_size=1)
        self.conv_V = nn.Conv1d(in_channels=in_channels, out_channels=in_channels*n_heads, groups=n_heads, kernel_size=1)
        self.conv_combine = nn.Conv1d(in_channels=in_channels*n_heads,out_channels=in_channels,kernel_size=1)
        # self.dropout1 = torch.nn.Dropout(p=0.5)
        # self.dropout2 = torch.nn.Dropout(p=0.5)
        # self.dropout3 = torch.nn.Dropout(p=0.5)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        # Derive parameters for pooling
        N, C, H_in, W_in = x.size()
        H_out, W_out = self.pooling_output_size
        kernel = s = H_in // H_out, W_in // W_out
        #padding = (H_out*s[0] - H_in) // 2, (W_out*s[1] - W_in)
        padding = (s[0] - 2)*H_out // 2, (s[1] - 2)*W_out // 2
        x_avg = F.avg_pool2d(x,kernel_size=kernel, stride=s, padding=padding)

        #x_avg = self.pool(x)
        N, C, H, W = x_avg.size()
        x_avg = x_avg.view(N, C, H * W)
        x_avg = x_avg.transpose(1, 2)  # Reshape to channel-wise vector at each x_avg[0,0,:]
        Q = self.conv_Q(x_avg)  # N x (H/r*W/r) x c*n_heads
        Q = Q.transpose(1,2).view(-1,C,self.n_heads,self.h_size) # Shape: N x C x n_head x h_size
        #Q = self.dropout1(Q)
        K = self.conv_K(x_avg)  # N x (H/r*W/r) x c
        K = K.transpose(1,2).view(-1, C, self.n_heads, self.h_size)
        #K = self.dropout2(K)
        # V = x_avg
        V = self.conv_V(x_avg) # The estimated scale that we should apply to each local neighborhood
        V = V.transpose(1,2).view(-1, C, self.n_heads, H * W)

        #score = torch.matmul(Q.transpose(1, 2), K)
        score = torch.einsum('...xhd,...yhd->...hxy',Q,K)
        score = F.softmax(score, dim=-1)
        #att_weights = torch.matmul(score, V.transpose(1, 2))  # att_weights = (C x C) x (C x (H*W)) = C x (H*W)
        weights = torch.einsum('...hcc,...chd->...hcd',score,V) # Shape: n_heads x C x (H*W)
        weights = weights.transpose(1,2).view(-1,C,self.n_heads*H*W)
        att_weights = self.conv_combine(weights.transpose(1,2))
        att_weights = self.dropout(att_weights)
        # => Re-balance the scale for each channel based on their importance relative to other channels

        # Repeat the attention weights by the stride of pooling layer
        # to transform weight_mask matching the shape of original input
        h_scale, w_scale = x.size(2) // self.pooling_output_size[0], x.size(3) // self.pooling_output_size[1]
        att_weights = att_weights.view(N,C,H,W)
        att_weights = F.interpolate(att_weights,scale_factor=(h_scale,w_scale),mode='nearest')
        # att_weights = att_weights.view(N, C, H * W, 1)
        # att_weights = att_weights.repeat(1, 1, 1, w_scale)
        # att_weights = att_weights.view(N, C, H, W * w_scale)
        # att_weights = att_weights.repeat(1, 1, 1, h_scale)
        # att_weights = att_weights.view(N, C, H * h_scale, W * w_scale)

        if att_weights.size() != x.size():
            att_weights = F.interpolate(att_weights, size=list(x.shape[2:]), mode='nearest')
        assert att_weights.size() == x.size()

        # Re-weight original input by weight mask
        return att_weights * x

class SceneUnderstandingModule(nn.Module):
    def __init__(self):
        super(SceneUnderstandingModule, self).__init__()
        total_ord_label = 68 # For NYU
        #total_ord_label = 90 # For uow
        self.channels = 512
        self.encoder = FullImageEncoder(self.channels)
        #total_K = (total_ord_label-1)*2 For probabilistic inference
        total_K = total_ord_label*2
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
        self.concat_process = nn.Sequential(
            nn.Dropout2d(p=0.5),
            ChannelwiseLocalAttention(pooling_output_size=(16,22),n_heads=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(512*5,self.channels,1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(self.channels,total_K,1),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(257,353)),
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
        decode_c = torch.sum((ord_c1 > 0.5), dim=1).view(-1, 1, H, W) # The one-label pixel corresponds to one rank
        # decode_c = torch.sum(ord_c1, dim=1).view(-1, 1, H, W)
       # Derive rank based on probabilistic
        if torch.cuda.is_available():
            decode_c = decode_c.cuda()

        return decode_c, ord_c1


class DORN(nn.Module):
    def __init__(self, output_size=(257, 353), channel=3, pretrained=True, freeze=True):
        super(DORN, self).__init__()

        self.output_size = output_size
        self.channel = channel
        #self.feature_extractor = resnet101(pretrained=pretrained)
        #self.feature_extractor = models.resnet18(pretrained=pretrained)
        #self.feature_extractor = resnet18(pretrained=pretrained)
        model = models.resnet18(pretrained=True)
        self.feature_extractor = torch.nn.Sequential(*(list(model.children())[:-2]))
        self.aspp_module = SceneUnderstandingModule()
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
