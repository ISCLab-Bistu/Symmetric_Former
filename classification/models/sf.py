# --------------------------------------------------------
# SF
# Copyright (c) 2022 ISCLab-Bistu
# Licensed under The MIT License [see LICENSE for details]
# Written by Shengjun Liang
# --------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.dw = torch.nn.Sequential(nn.Conv2d(
                                     in_channels,
                                     out_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     groups=out_channels,
                                     padding=padding),
        )

    def forward(self, x):
        x = self.dw(x)
        return x


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.GELU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class Mlp_ghost(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.hidden_features = hidden_features
        self.in_features = in_features
        self.primary_conv_i = nn.Sequential(
            nn.Conv2d(in_features, hidden_features//2, 1, 1),
            nn.GELU()
        )
        self.cheap_operation_i = nn.Sequential(
            nn.Conv2d(hidden_features // 2, hidden_features // 2, 3, 1, 3 // 2, groups=hidden_features // 2),
            nn.GELU()
        )

        self.primary_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features // 2, 3, 1, 1, groups=hidden_features // 2),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(hidden_features // 2, out_features, 1, 1),
        )

    def forward(self, x):
        B, N, c = x.shape
        res = int(np.sqrt(N))

        x = x.permute(0, 2, 1).view(B, c, res, res)

        x1 = self.primary_conv_i(x) # *2
        x2 = self.cheap_operation_i(x1)# *2
        x = torch.cat([x1, x2], dim=1)# *4

        x = channel_shuffle(x, 2)
        # x = self.norm(x)

        x = x1 + self.primary_conv(x)# *1/2
        x = self.cheap_operation(x)# *1/2
        # out = torch.cat([x1, x2], dim=1)# *1
        out = x.view(B, c, N).permute(0, 2, 1)
        return out

    def flops(self, H, W):
        flops = H * W * self.in_features * self.hidden_features // 2 + \
            9 * H * W * self.hidden_features // 2 + \
            9 * H * W * self.hidden_features // 2 + \
            H * W * self.in_features * self.hidden_features // 2
        return flops


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.hidden_features = hidden_features
        self.in_features = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.enc = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        B, N, c = x.shape
        res = int(np.sqrt(N))
        x = x.permute(0, 2, 1).view(B, c, res, res)
        x = self.enc(x)
        x = x.view(B, c, N).permute(0, 2, 1)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def flops(self, H, W):
        flops = 2 * H * W * self.in_features * self.hidden_features
        return flops


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., inverter=True, ori=False, expand=False):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        # self.dif_rate = 6 if num_heads % 6 == 0 else 3
        # self.dif = False
        self.acspa_use = False
        self.position = True
        self.inverter = inverter
        if self.inverter:
            self.inverter_rate = 1.75
            self.inv_head_dim = int(head_dim*self.inverter_rate)
            self.head_dim = int(head_dim*self.inverter_rate)
        else:
            self.inv_head_dim = head_dim
            self.head_dim = head_dim
        # define a parameter table of relative position bias
        if self.position:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, self.num_heads*(self.head_dim * 2 + self.inv_head_dim), bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        if self.acspa_use:
            self.down1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
            self.down2 = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)

            self.vfc1 = nn.Linear(3, 16, bias=False)
            self.vfc2 = nn.Linear(16, 3, bias=False)
            self.soft = nn.Softmax(1)

        self.ori = ori # true
        self.ori_dim = 0

        if self.ori:
            self.ori_dim = self.dim
            self.linal_trans = SeparableConv2d(dim,
                                       dim, kernel_size=3, padding=1, bias=True)

        self.Lepe = True # true
        if self.Lepe:
            self.lev = SeparableConv2d(self.num_heads*self.inv_head_dim,
                                       self.inv_head_dim*self.num_heads, kernel_size=3, padding=1, bias=True)

        self.proj = nn.Linear(self.num_heads*self.inv_head_dim+self.ori_dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)
        if self.position:
            trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        res = int(np.sqrt(N))
        # x_ee = x_ee.permute(0, 2, 3, 1).reshape(B_, N, self.h_o)
        # B_, N, 3, hn, fn
        qkv = self.qkv(x).reshape(B_, N, self.num_heads, (self.head_dim * 2 + self.inv_head_dim)).permute(0, 2, 1, 3)
        # 3, B_, hn, N, fn
        q, k, v = qkv[:, :, :, :self.head_dim], qkv[:, :, :, self.head_dim:self.head_dim*2],\
                  qkv[:, :, :, self.head_dim*2:]  # make torchscript happy (cannot use tensor as tuple)

        if self.Lepe:
            # print('q', v.shape, B_, self.num_heads * self.inv_head_dim, res, res)
            lepe = self.lev(v.permute(0, 1, 3, 2).reshape(B_, self.num_heads * self.inv_head_dim, res, res)).\
                view(B_, self.inv_head_dim * self.num_heads, res * res).permute(0, 2, 1)
        q = q * self.scale
        # W*B, 49, head, feature
        if self.acspa_use:
            q, k = self.acspa(q, k)

        attn = (q @ k.transpose(-2, -1))
        if self.position:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        # l_d = self.Adown(attn).view(B_, self.num_heads)
        # l_e = self.buildin_mlp(l_d).view(B_, self.num_heads, 1, 1)
        # attn = attn*l_e
        attn = self.attn_drop(attn)
        x_ = (attn @ v).transpose(1, 2).reshape(B_, N, self.inv_head_dim*self.num_heads)

        if self.Lepe:
            # print(self.dim, x.shape, lepe.shape)
            x_ = x_ + lepe

        if self.ori:
            # x_expand = self.linal_trans(x)

            res = int(np.sqrt(N))
            x_e = self.linal_trans(x.permute(0, 2, 1).view(B_, C, res, res)).\
                view(B_, C, res * res).permute(0, 2, 1)
            x_ = torch.cat([x_, x_e], 2)
        x = self.proj(x_)
        x = self.proj_drop(x)
        # self.flops(1)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def acspa(self, q, k):
        N_, hn, s, fn = q.shape
        # W*B, head, 49, feature
        qf = q.permute(0, 1, 3, 2).view(N_, hn*fn, self.window_size[0], self.window_size[1])
        q_1 = self.down1(qf)
        q_2 = self.down2(qf)

        kf = k.permute(0, 1, 3, 2).view(N_, hn*fn, self.window_size[0], self.window_size[1])
        k_1 = self.down1(kf)
        k_2 = self.down2(kf)

        q_spe = torch.stack([qf, q_1, q_2], 2).view(N_, hn, fn, 3, self.window_size[0], self.window_size[1])\
            .mean(2).mean(3).mean(3)
        k_spe = torch.stack([kf, k_1, k_2], 2).view(N_, hn, fn, 3, self.window_size[0], self.window_size[1])\
            .mean(2).mean(3).mean(3)
        # n_, hn, 3
        # print(q_spe.shape, k_spe.shape)
        q_scfm = self.soft(self.vfc2(self.vfc1(q_spe)))
        k_scfm = self.soft(self.vfc2(self.vfc1(k_spe)))

        # print(q_scfm.shape, k_scfm.shape, qf.shape, hn, fn)
        q = qf.view(N_, hn, fn, self.window_size[0], self.window_size[1])\
            *q_scfm[:, :, 0].unsqueeze(2).unsqueeze(2).unsqueeze(2)+ \
            q_1.view(N_, hn, fn, self.window_size[0], self.window_size[1])\
            *q_scfm[:, :, 1].unsqueeze(2).unsqueeze(2).unsqueeze(2)+ \
            q_2.view(N_, hn, fn, self.window_size[0], self.window_size[1])\
            *q_scfm[:, :, 2].unsqueeze(2).unsqueeze(2).unsqueeze(2)

        k = kf.view(N_, hn, fn, self.window_size[0], self.window_size[1])\
            *k_scfm[:, :, 0].unsqueeze(2).unsqueeze(2).unsqueeze(2)+ \
            k_1.view(N_, hn, fn, self.window_size[0], self.window_size[1])\
            *k_scfm[:, :, 1].unsqueeze(2).unsqueeze(2).unsqueeze(2)+ \
            k_2.view(N_, hn, fn, self.window_size[0], self.window_size[1])\
            *k_scfm[:, :, 2].unsqueeze(2).unsqueeze(2).unsqueeze(2)

        return q.permute(0, 1, 3, 4, 2).view(N_, hn, s, fn), \
               k.permute(0, 1, 3, 4, 2).view(N_, hn, s, fn)

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        # flops = 0
        # qkv = self.qkv(x)
        # print('window size', N)
        flops1 = N * self.dim * 3 * self.head_dim * self.num_heads
        # attn = (q @ k.transpose(-2, -1))
        flops2 = self.num_heads//3 * N * self.head_dim * N
        #  x = (attn @ v)
        flops3 = self.num_heads//3 * N * N * self.head_dim
        # x = self.proj(x)
        flops4 = N * self.dim * (self.head_dim * self.num_heads + self.ori_dim)
        # print('dim', self.dim, 'hdim', self.head_dim, 'nh', self.num_heads, self.dim == self.head_dim*self.num_heads)
        # if self.ori:
        #     # x_expand = self.linal_trans(x)
        #     flops += N * self.num_heads * self.head_dim * self.num_heads * self.head_dim * (self.ori_rate - 1)
        if self.Lepe:
            # lepe = self.lev(v)
            flops5 = N * 2 * 9 *\
                     self.num_heads * self.head_dim
            if self.ori:
                flops5 += N * 2 * 9 *\
                     self.dim
            flops = flops1 + flops2 + flops3 + flops4 + flops5
            # print('{0:.3f}, {1:.3f}, {2:.3f}, {3:.3f}, {4:.3f}'.format(flops1 / flops, flops2 / flops, flops3 / flops, flops4 / flops, flops5 / flops))
        else:
            flops = flops1 + flops2 + flops3 + flops4
            # print('{0:.3f}, {1:.3f}, {2:.3f}, {3:.3f}'.format(flops1 / flops, flops2 / flops, flops3 / flops, flops4 / flops))
        return flops


class SFBlock(nn.Module):
    r""" SY Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, expand=False, stage=0, inverter=True, ori=True):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, expand=expand, inverter=True, ori=True)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio[stage])
        # if stage <= 3: # stage == 2 or 3
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp_ghost(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # else:
        # self.norm2 = norm_layer(dim)
        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # H, W = self.input_resolution
        # self.window_size = H
        # self.shift_size = 0
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += self.mlp.flops(H, W)
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic SY layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False, expand=False, stage=0,
                 inverter=True, ori=True):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            print(i, ori)
            if ori and i%2==0:
                ori_ = True
            else:
                ori_ = False
            blo = SFBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 expand=expand,
                                 stage=stage,
                                 inverter=inverter,
                                 ori=ori_)
            self.blocks.append(blo)

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SF(nn.Module):
    r""" SY
        A PyTorch impl of : `SY: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each SY layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            if i_layer == 0:
                expand = False
                inverter=False
                ori = True
            else:
                expand = False
                inverter=True
                ori = True

            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               expand=expand,
                               stage=i_layer,
                               ori=ori,
                               inverter=inverter)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
