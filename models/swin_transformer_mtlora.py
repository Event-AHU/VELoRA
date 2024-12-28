# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch
from torch import Tensor
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .lora import MTLoRALinear
import math
try:
    import os
    import sys

    kernel_path = os.path.abspath(os.path.join('..'))
    sys.path.append(kernel_path)
    from kernels.window_process.window_process import WindowProcess, WindowProcessReverse

except:
    WindowProcess = None
    WindowProcessReverse = None
    print("[Warning] Fused window process have not been installed. Please refer to get_started.md for installation.")


class CompatLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input: Tensor, x_tasks: dict = None) -> Tensor:
        return super().forward(input), None


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., lora=False, tasks=None, mtlora=None, layer_idx=0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        if mtlora.FC1_ENABLED:
            self.fc1 = MTLoRALinear(in_features, hidden_features, r=mtlora.R_PER_TASK_LIST[layer_idx] if lora else 0,
                                    lora_shared_scale=mtlora.SHARED_SCALE[layer_idx], lora_task_scale=mtlora.SCALE_PER_TASK_LIST[layer_idx], lora_dropout=mtlora.DROPOUT[layer_idx], tasks=(
                                        tasks if (lora or mtlora.INTERMEDIATE_SPECIALIZATION) else None),
                                    trainable_scale_shared=mtlora.TRAINABLE_SCALE_SHARED, trainable_scale_per_task=mtlora.TRAINABLE_SCALE_PER_TASK, shared_mode=mtlora.SHARED_MODE)
        else:
            self.fc1 = CompatLinear(in_features, hidden_features)
        self.act = act_layer()
        if mtlora.FC2_ENABLED:
            self.fc2 = MTLoRALinear(hidden_features, out_features, r=mtlora.R_PER_TASK_LIST[layer_idx] if lora else 0,
                                    lora_shared_scale=mtlora.SHARED_SCALE[layer_idx], lora_task_scale=mtlora.SCALE_PER_TASK_LIST[layer_idx], lora_dropout=mtlora.DROPOUT[layer_idx], tasks=(
                                        tasks if (lora or mtlora.INTERMEDIATE_SPECIALIZATION) else None),
                                    trainable_scale_shared=mtlora.TRAINABLE_SCALE_SHARED, trainable_scale_per_task=mtlora.TRAINABLE_SCALE_PER_TASK, shared_mode=mtlora.SHARED_MODE)
        else:
            self.fc2 = CompatLinear(hidden_features, out_features)
        self.tasks = tasks
        self.drop = nn.Dropout(drop)

    def forward(self, x, x_tasks=None):
        x, fc1_lora_tasks = self.fc1(x, x_tasks)
        x = self.act(x)
        x = self.drop(x)
        if fc1_lora_tasks is not None:
            for task in self.tasks:
                fc1_lora_tasks[task] = self.act(fc1_lora_tasks[task])
                fc1_lora_tasks[task] = self.drop(fc1_lora_tasks[task])
        x, fc2_lora_tasks = self.fc2(x, fc1_lora_tasks)
        x = self.drop(x)
        if fc2_lora_tasks is not None:
            for task in self.tasks:
                fc2_lora_tasks[task] = self.drop(fc2_lora_tasks[task])
        return x, fc2_lora_tasks


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    
    x = x.view(B, H // window_size, window_size,
               W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous(
    ).view(-1, window_size, window_size, C)
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
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
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

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., lora=False, tasks=None, mtlora=None, layer_idx=0):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - \
            coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - \
            1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index",
                             relative_position_index)
        if mtlora.QKV_ENABLED:
            self.qkv = MTLoRALinear(dim, dim * 3, r=mtlora.R_PER_TASK_LIST[layer_idx] if lora else 0,
                                    lora_shared_scale=mtlora.SHARED_SCALE[layer_idx], lora_task_scale=mtlora.SCALE_PER_TASK_LIST[layer_idx], lora_dropout=mtlora.DROPOUT[layer_idx], tasks=None, bias=qkv_bias,
                                    trainable_scale_shared=mtlora.TRAINABLE_SCALE_SHARED, trainable_scale_per_task=mtlora.TRAINABLE_SCALE_PER_TASK, shared_mode=mtlora.SHARED_MODE)
        else:
            self.qkv = CompatLinear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        if mtlora.PROJ_ENABLED:
            self.proj = MTLoRALinear(dim, dim, r=mtlora.R_PER_TASK_LIST[layer_idx] if lora else 0,
                                     lora_shared_scale=mtlora.SHARED_SCALE[layer_idx], lora_task_scale=mtlora.SCALE_PER_TASK_LIST[layer_idx], lora_dropout=mtlora.DROPOUT[layer_idx], tasks=(
                tasks if (lora or mtlora.INTERMEDIATE_SPECIALIZATION) else None),
                trainable_scale_shared=mtlora.TRAINABLE_SCALE_SHARED, trainable_scale_per_task=mtlora.TRAINABLE_SCALE_PER_TASK, shared_mode=mtlora.SHARED_MODE)
        else:
            self.proj = CompatLinear(dim, dim)

        self.tasks = tasks
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv, _ = self.qkv(x)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, C //
                          self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x, x_proj_lora_tasks = self.proj(x)
        x = self.proj_drop(x)
        if x_proj_lora_tasks is not None:
            for task in self.tasks:
                x_proj_lora_tasks[task] = self.proj_drop(
                    x_proj_lora_tasks[task])
        return x, x_proj_lora_tasks

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

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
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fused_window_process=False, lora=False, tasks=None, mtlora=None, layer_idx=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.tasks = tasks
        self.lora = lora
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, lora=lora, tasks=tasks, mtlora=mtlora, layer_idx=layer_idx)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop, lora=lora, tasks=tasks, mtlora=mtlora, layer_idx=layer_idx)

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

            # nW, window_size, window_size, 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1,
                                             self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        self.fused_window_process = fused_window_process

    def forward(self, x):
       
        H, W = self.input_resolution
        B, L, C = x.shape
        
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = torch.roll(
                    x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                # partition windows
                # nW*B, window_size, window_size, C
                x_windows = window_partition(shifted_x, self.window_size)
            else:
                x_windows = WindowProcess.apply(
                    x, B, H, W, C, -self.shift_size, self.window_size)
        else:
            shifted_x = x
            # partition windows
            # nW*B, window_size, window_size, C
            x_windows = window_partition(shifted_x, self.window_size)

        # nW*B, window_size*window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        # nW*B, window_size*window_size, C
        attn_windows, attn_windows_lora_tasks = self.attn(
            x_windows, mask=self.attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1,
                                         self.window_size, self.window_size, C)

        # reverse cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = window_reverse(
                    attn_windows, self.window_size, H, W)  # B H' W' C
                x = torch.roll(shifted_x, shifts=(
                    self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = WindowProcessReverse.apply(
                    attn_windows, B, H, W, C, self.shift_size, self.window_size)
        else:
            shifted_x = window_reverse(
                attn_windows, self.window_size, H, W)  # B H' W' C
            x = shifted_x
        if attn_windows_lora_tasks is not None:
            for task in self.tasks:
                attn_windows_lora_tasks[task] = attn_windows_lora_tasks[task].view(
                    -1, self.window_size, self.window_size, C)
                attn_windows_lora_tasks[task] = window_reverse(
                    attn_windows_lora_tasks[task], self.window_size, H, W)
                if self.shift_size > 0:
                    attn_windows_lora_tasks[task] = torch.roll(attn_windows_lora_tasks[task], shifts=(
                        self.shift_size, self.shift_size), dims=(1, 2))
                attn_windows_lora_tasks[task] = attn_windows_lora_tasks[task].view(
                    B, H * W, C)
                attn_windows_lora_tasks[task] = shortcut + \
                    self.drop_path(attn_windows_lora_tasks[task])
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        # FFN
        mlp_result, mlp_lora_tasks = self.mlp(
            self.norm2(x), {task: self.norm2(attn_windows_lora_tasks[task]) for task in self.tasks} if attn_windows_lora_tasks is not None else None)

        if mlp_lora_tasks is None:
            return x + self.drop_path(mlp_result), None
        else:
            if attn_windows_lora_tasks is None:
                for task in self.tasks:
                    mlp_lora_tasks[task] = self.drop_path(mlp_lora_tasks[task])
            else:
                for task in self.tasks:
                    mlp_lora_tasks[task] = attn_windows_lora_tasks[task] + \
                        self.drop_path(mlp_lora_tasks[task])
            return x + self.drop_path(mlp_result), mlp_lora_tasks

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
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
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

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm, layer_idx=0, mtlora=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        if mtlora.DOWNSAMPLER_ENABLED:
            self.reduction = MTLoRALinear(4 * dim, 2 * dim, r=mtlora.R_PER_TASK_LIST[layer_idx],
                                          lora_shared_scale=mtlora.SHARED_SCALE[layer_idx], lora_task_scale=mtlora.SCALE_PER_TASK_LIST[layer_idx], lora_dropout=mtlora.DROPOUT[layer_idx], tasks=None, bias=False,
                                          trainable_scale_shared=mtlora.TRAINABLE_SCALE_SHARED, trainable_scale_per_task=mtlora.TRAINABLE_SCALE_PER_TASK, shared_mode=mtlora.SHARED_MODE)
        else:
            self.reduction = CompatLinear(4 * dim, 2 * dim, bias=False)

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
        x, _ = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

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
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 fused_window_process=False, tasks=None, mtlora=None, layer_idx=0):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.tasks = tasks

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (
                                     i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(
                                     drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 fused_window_process=fused_window_process,
                                 lora=(i == depth - 1),
                                 tasks=tasks,
                                 mtlora=mtlora,
                                 layer_idx=layer_idx)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, norm_layer=norm_layer, layer_idx=layer_idx, mtlora=mtlora)
        else:
            self.downsample = None

    def forward(self, x):
        
        for blk in self.blocks:
            x, tasks_lora = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
            if tasks_lora is not None:
                for task in self.tasks:
                    tasks_lora[task] = self.downsample(tasks_lora[task])
        return x, tasks_lora

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
        patches_resolution = [img_size[0] //
                              patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
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
        flops = Ho * Wo * self.embed_dim * self.in_chans * \
            (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SwinTransformerMTLoRA(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
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
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, fused_window_process=False,
                 basic_layer=BasicLayer, tasks=None, mtlora=None, ** kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.tasks = tasks
        self.mtlora = mtlora

        # Print lora params:
        if mtlora is not None:
            print("\nMTLoRA params:")
            print(mtlora)

        # split image into non-overlapping patches
        # self.patch_embed = PatchEmbed(
        #     img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
        #     norm_layer=norm_layer if self.patch_norm else None)
        # num_patches = self.patch_embed.num_patches
        # patches_resolution = self.patch_embed.patches_resolution
        # self.patches_resolution = patches_resolution

        # absolute position embedding
        # if self.ape:
        #     self.absolute_pos_embed = nn.Parameter(
        #         torch.zeros(1, num_patches, embed_dim))
        #     trunc_normal_(self.absolute_pos_embed, std=.02)

        # self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                 sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        
        # for i_layer in range(self.num_layers):
        #     layer = basic_layer(dim=int(embed_dim * 2 ** i_layer),  #96 192
        #                         input_resolution=(patches_resolution[0] // (2 ** i_layer),
        #                                           patches_resolution[1] // (2 ** i_layer)), #(112,112) (56,56)
        #                         depth=depths[i_layer], #2
        #                         num_heads=num_heads[i_layer], #3 6 12 24
        #                         window_size=window_size, #7
        #                         mlp_ratio=self.mlp_ratio, #4.0
        #                         qkv_bias=qkv_bias, qk_scale=qk_scale,
        #                         drop=drop_rate, attn_drop=attn_drop_rate,
        #                         drop_path=dpr[sum(depths[:i_layer]):sum(
        #                             depths[:i_layer + 1])],  #[0.0, 0.0181818176060915]
        #                         norm_layer=norm_layer,
        #                         downsample=PatchMerging if (
        #                             i_layer < self.num_layers - 1) else None,
        #                         use_checkpoint=use_checkpoint,
        #                         fused_window_process=fused_window_process,
        #                         tasks=tasks,
        #                         mtlora=self.mtlora,
        #                         layer_idx=i_layer)
            
        #     self.layers.append(layer)
        
        layer = basic_layer(dim=int(768),
                            input_resolution=(18,18),
                            depth=depths[3],
                            num_heads=num_heads[3],
                            window_size=9,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dpr[sum(depths[:3]):sum(
                                depths[:3+ 1])],
                            norm_layer=norm_layer,
                            downsample=PatchMerging if (
                                    3 < self.num_layers - 1) else None,
                            use_checkpoint=use_checkpoint,
                            fused_window_process=fused_window_process,
                            tasks=tasks,
                            mtlora=self.mtlora,
                            layer_idx=3)
        self.layers.append(layer)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(
            self.num_features, num_classes) if num_classes > 0 else nn.Identity()

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

    def forward_features(self, x, return_stages=False, flatten_ft=False):
       
        #x = self.patch_embed(x)   # 输入[4,3,448,448] 出来【4，12544，96】嵌入先撤
        #if self.ape:
         #   x = x + self.absolute_pos_embed
        #x = self.pos_drop(x)
       
        
        if return_stages:
            out = []
        i = 0
        
        for layer in self.layers:
           x, tasks_lora = layer(x)
           if tasks_lora is None:
               tasks_lora = {task: x for task in self.tasks}
           if return_stages:
               out.append((x, tasks_lora))
           i = i + 1
        #假设只走一个layer
       
        #x,tasks_lora =  self.layers[0](x)
        # if tasks_lora is None:
        #     tasks_lora = {task: x for task in self.tasks}
        # if return_stages:
        #     out.append((x, tasks_lora))
               
                
       
        if return_stages:
            return out       #out len=4 
        else:
            if flatten_ft:
                x = self.avgpool(x.transpose(1, 2))  # B C 1
                x = torch.flatten(x, 1)
            return x

    def forward(self, x, return_stages=False, flatten_ft=False):
       
        x = self.forward_features(x, return_stages, flatten_ft)
        
        x = self.head(x)  #torch.Size([4, 3136, 192])  
        return x

    def flops(self, images=None, logger=None, detailed=False):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * \
            self.patches_resolution[0] * \
            self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops