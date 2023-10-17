## Our PoseFormer model was revised from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

import math
import logging
import numpy as np
from functools import partial
from collections import OrderedDict
from einops import rearrange, repeat
from scipy.interpolate import interp1d

import torch
import torch.nn as nn
import torch_dct as dct
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

import pywt


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class FreqMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        # Initialize wavelet transforms
        self.dwt = DWTForward(J=1, wave='haar')
        self.idwt = DWTInverse(wave='haar')

    def forward(self, x):
        b, f, _ = x.shape

        # Reshape x to fit the expected input shape for DWTForward
        x_reshaped = x.unsqueeze(1)  # Shape: [batch, 1, features, length]

        # Use wavelet transform to replace DCT
        coeffs = self.dwt(x_reshaped)
        x_low = coeffs[0].squeeze(1)  # remove the singleton channel dimension
        x_high = coeffs[1][0].squeeze(1)  # remove the singleton channel dimension

        # Convert it to NumPy array
        x_low_np = x_low.cpu().detach().numpy()

        # Initialize an empty array with the target size
        x_interp_np = np.zeros((b, f, x.shape[2]))  # Adjusted to use shape variables

        # Perform interpolation
        for i in range(b):
            for j in range(f):
                f_interp = interp1d(np.linspace(0, 1, x_low.shape[-1]), x_low_np[i, j, :], kind='linear')
                x_interp_np[i, j, :] = f_interp(np.linspace(0, 1, x.shape[-1]))

        # Convert the result back to PyTorch tensor and move to the same device as `x`
        x_low_interp = torch.tensor(x_interp_np).float().to(x.device)

        # Perform interpolation
        for i in range(x_low_np.shape[0]):
            for j in range(x_low_np.shape[1]):
                f = interp1d(np.linspace(0, 1, x_low[0].shape[-1]), x_low_np[i, j, :], kind='linear')
                x_interp_np[i, j, :] = f(np.linspace(0, 1, x.shape[-1]))

        # Convert the result back to PyTorch tensor and move to the same device as `x`
        x_low[0] = torch.tensor(x_interp_np).float().to(x.device)

        # Perform the original MLP operations
        x_low[0] = self.fc1(x_low[0])
        x_low[0] = self.act(x_low[0])
        x_low[0] = self.drop(x_low[0])
        x_low[0] = self.fc2(x_low[0])
        x_low[0] = self.drop(x_low[0])

        # Reshape x_low back to [batch, features, length]
        x_low = x_low.squeeze(3)

        # Use inverse wavelet transform
        x_reconstructed = self.idwt((x_low.unsqueeze(3), [x_high.unsqueeze(3)]))
        x_reconstructed = torch.tensor(x_reconstructed).to(x.device).float()

        # Remove the singleton dimension and adjust shape
        x_reconstructed = x_reconstructed.squeeze(1)

        # # Adjust size to be consistent with original x
        # if x_reconstructed.shape[-1] > x.shape[-1]:
        #     x_reconstructed = x_reconstructed[..., :x.shape[-1]]

        return x_reconstructed


# class FreqMlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)
#
#     def forward(self, x):
#         b, f, _ = x.shape
#         x = dct.dct(x.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         x = dct.idct(x.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
#         return x


# class Attention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
#         self.scale = qk_scale or head_dim ** -0.5
#
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#     def forward(self, x):
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=16, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_linear = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)

        # Compute attention score
        attn_score = F.relu(q.unsqueeze(2) + k.unsqueeze(1))  # Shape: [B, N, N, C]

        # Compute attention weights
        attn = F.softmax(attn_score.sum(dim=-1), dim=-1)  # Summing over the last dimension to get shape [B, N, N]
        attn = self.attn_drop(attn)

        # Compute output
        x = (attn.unsqueeze(-1) * v.unsqueeze(1)).sum(dim=2)  # Shape: [B, N, C]
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm3 = norm_layer(dim)
        self.mlp2 = FreqMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        b, f, c = x.shape
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x1 = x[:, :f//2] + self.drop_path(self.mlp1(self.norm2(x[:, :f//2])))
        x2 = x[:, f//2:] + self.drop_path(self.mlp2(self.norm3(x[:, f//2:])))
        return torch.cat((x1, x2), dim=1)

class PoseTransformer(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=16, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=None):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio * num_joints   #### temporal embed_dim is num_joints * spatial embedding dim ratio
        out_dim = num_joints * 3     #### output dimension is num_joints * 3

        ### spatial patch embedding
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))

        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.Freq_embedding = nn.Linear(in_chans*num_joints, embed_dim)


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)

        ####### A easy way to implement weighted mean
        self.weighted_mean = torch.nn.Conv1d(in_channels=num_frame, out_channels=1, kernel_size=1)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim , out_dim),
        )


    def Spatial_forward_features(self, x):
        b, _, f, p = x.shape  ##### b is batch size, f is number of frames, p is number of joints
        x = rearrange(x, 'b c f p  -> (b f) p  c', )

        x = self.Spatial_patch_to_embedding(x)
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)

        for blk in self.Spatial_blocks:
            x = blk(x)

        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) w c -> b f (w c)', f=f)
        return x


    def forward_features(self, x):
        b = x.shape[0]
        x += self.Temporal_pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)

        x = self.Temporal_norm(x)
        ##### x size [b, f, emb_dim], then take weighted mean on frame dimension, we only predict 3D pose of the center frame
        x = self.weighted_mean(x)
        x = x.view(b, 1, -1)
        return x


    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        b, _, _, p = x.shape
        ### now x is [batch_size, 2 channels, receptive frames, joint_num], following image data
        x = self.Spatial_forward_features(x)
        x = self.forward_features(x)
        x = self.head(x)

        x = x.view(b, 1, p, -1)

        return x

