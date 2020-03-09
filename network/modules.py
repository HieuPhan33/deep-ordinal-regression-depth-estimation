import torch
from torch import nn
import math
import torch.nn.functional as F

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

class AugmentedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dk, dv, Nh, shape=0, relative=False, stride=1):
        super(AugmentedConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dk = dk
        self.dv = dv
        self.Nh = Nh
        self.shape = shape
        self.relative = relative
        self.stride = stride
        self.padding = (self.kernel_size - 1) // 2

        assert self.Nh != 0, "integer division or modulo by zero, Nh >= 1"
        assert self.dk % self.Nh == 0, "dk should be divided by Nh. (example: out_channels: 20, dk: 40, Nh: 4)"
        assert self.dv % self.Nh == 0, "dv should be divided by Nh. (example: out_channels: 20, dv: 4, Nh: 4)"
        assert stride in [1, 2], str(stride) + " Up to 2 strides are allowed."

        self.conv_out = nn.Conv2d(self.in_channels, self.out_channels - self.dv, self.kernel_size, stride=stride, padding=self.padding)

        self.qkv_conv = nn.Conv2d(self.in_channels, 2 * self.dk + self.dv, kernel_size=self.kernel_size, stride=stride, padding=self.padding)

        self.attn_out = nn.Conv2d(self.dv, self.dv, kernel_size=1, stride=1)

        if self.relative:
            self.key_rel_w = nn.Parameter(torch.randn((2 * self.shape - 1, dk // Nh), requires_grad=True))
            self.key_rel_h = nn.Parameter(torch.randn((2 * self.shape - 1, dk // Nh), requires_grad=True))

    def forward(self, x):
        # Input x
        # (batch_size, channels, height, width)
        # batch, _, height, width = x.size()

        # conv_out
        # (batch_size, out_channels, height, width)
        conv_out = self.conv_out(x)
        batch, _, height, width = conv_out.size()

        # flat_q, flat_k, flat_v
        # (batch_size, Nh, height * width, dvh or dkh)
        # dvh = dv / Nh, dkh = dk / Nh
        # q, k, v
        # (batch_size, Nh, height, width, dv or dk)
        flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(x, self.dk, self.dv, self.Nh)
        logits = torch.matmul(flat_q.transpose(2, 3), flat_k)
        if self.relative:
            h_rel_logits, w_rel_logits = self.relative_logits(q)
            logits += h_rel_logits
            logits += w_rel_logits
        weights = F.softmax(logits, dim=-1)

        # attn_out
        # (batch, Nh, height * width, dvh)
        attn_out = torch.matmul(weights, flat_v.transpose(2, 3))
        attn_out = torch.reshape(attn_out, (batch, self.Nh, self.dv // self.Nh, height, width))
        # combine_heads_2d
        # (batch, out_channels, height, width)
        attn_out = self.combine_heads_2d(attn_out)
        attn_out = self.attn_out(attn_out)
        return torch.cat((conv_out, attn_out), dim=1)

    def compute_flat_qkv(self, x, dk, dv, Nh):
        qkv = self.qkv_conv(x)
        N, _, H, W = qkv.size()
        q, k, v = torch.split(qkv, [dk, dk, dv], dim=1)
        q = self.split_heads_2d(q, Nh)
        k = self.split_heads_2d(k, Nh)
        v = self.split_heads_2d(v, Nh)

        dkh = dk // Nh
        q *= dkh ** -0.5
        flat_q = torch.reshape(q, (N, Nh, dk // Nh, H * W))
        flat_k = torch.reshape(k, (N, Nh, dk // Nh, H * W))
        flat_v = torch.reshape(v, (N, Nh, dv // Nh, H * W))
        return flat_q, flat_k, flat_v, q, k, v

    def split_heads_2d(self, x, Nh):
        batch, channels, height, width = x.size()
        ret_shape = (batch, Nh, channels // Nh, height, width)
        split = torch.reshape(x, ret_shape)
        return split

    def combine_heads_2d(self, x):
        batch, Nh, dv, H, W = x.size()
        ret_shape = (batch, Nh * dv, H, W)
        return torch.reshape(x, ret_shape)

    def relative_logits(self, q):
        B, Nh, dk, H, W = q.size()
        q = torch.transpose(q, 2, 4).transpose(2, 3)

        rel_logits_w = self.relative_logits_1d(q, self.key_rel_w, H, W, Nh, "w")
        rel_logits_h = self.relative_logits_1d(torch.transpose(q, 2, 3), self.key_rel_h, W, H, Nh, "h")

        return rel_logits_h, rel_logits_w

    def relative_logits_1d(self, q, rel_k, H, W, Nh, case):
        rel_logits = torch.einsum('bhxyd,md->bhxym', q, rel_k)
        rel_logits = torch.reshape(rel_logits, (-1, Nh * H, W, 2 * W - 1))
        rel_logits = self.rel_to_abs(rel_logits)

        rel_logits = torch.reshape(rel_logits, (-1, Nh, H, W, W))
        rel_logits = torch.unsqueeze(rel_logits, dim=3)
        rel_logits = rel_logits.repeat((1, 1, 1, H, 1, 1))

        if case == "w":
            rel_logits = torch.transpose(rel_logits, 3, 4)
        elif case == "h":
            rel_logits = torch.transpose(rel_logits, 2, 4).transpose(4, 5).transpose(3, 5)
        rel_logits = torch.reshape(rel_logits, (-1, Nh, H * W, H * W))
        return rel_logits

    def rel_to_abs(self, x):
        B, Nh, L, _ = x.size()

        col_pad = torch.zeros((B, Nh, L, 1)).to(x)
        x = torch.cat((x, col_pad), dim=3)

        flat_x = torch.reshape(x, (B, Nh, L * 2 * L))
        flat_pad = torch.zeros((B, Nh, L - 1)).to(x)
        flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)

        final_x = torch.reshape(flat_x_padded, (B, Nh, L + 1, 2 * L - 1))
        final_x = final_x[:, :, :L, L - 1:]
        return final_x


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