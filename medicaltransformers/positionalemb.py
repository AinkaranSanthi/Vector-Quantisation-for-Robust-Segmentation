import math
import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss, MSELoss
def to_2tuple(x):
    if isinstance(x, int):
        return (x, x)

def to_3tuple(x):
    if isinstance(x, int):
        return (x, x, x)

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert (H == self.img_size[0]),  f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
        assert (W == self.img_size[1]), f"Input image width ({W}) doesn't match model ({self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

class PatchEmbed3d(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_3tuple(img_size)
        patch_size = to_3tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, Z, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x
import math
from typing import Tuple

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """
    Applies a positional encoding to a tensor with shape (batch_size x seq_len x embed_dim).
    The positional encoding is computed as follows:
        PE(pos,2i) = sin(pos/10000^(2i/dmodel))
        PE(pos,2i+1) = cos(pos/10000^(2i/dmodel))
        where pos = position, pos in [0, seq_len)
        dmodel = data embedding dimension = embed_dim
        i = dimension index, i in [0, embed_dim)
    Reference: "Attention Is All You Need" https://arxiv.org/abs/1706.03762
    Implementation Reference: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, embed_dim: int, seq_len: int = 1024) -> None:
        super().__init__()
        pe = torch.zeros(seq_len, embed_dim, dtype=torch.float)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-(math.log(10000.0)) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.pe.size(1) >= x.size(1), (
            "Cannot apply position encoding of size "
            + f"{self.pe.size()} when input has size {x.size()}"
        )
        return x + self.pe[:, : x.size(1), :]


class SpatioTemporalClsPositionalEncoding(nn.Module):
    """
    Add a cls token and apply a spatiotemporal encoding to a tensor.
    """

    def __init__(
        self,
        embed_dim: int,
        patch_embed_shape: Tuple[int, int, int],
        sep_pos_embed: bool = False,
        has_cls: bool = True,
    ) -> None:
        """
        Args:
            embed_dim (int): Embedding dimension for input sequence.
            patch_embed_shape (Tuple): The number of patches in each dimension
                (T, H, W) after patch embedding.
            sep_pos_embed (bool): If set to true, one positional encoding is used for
                spatial patches and another positional encoding is used for temporal
                sequence. Otherwise, only one positional encoding is used for all the
                patches.
            has_cls (bool): If set to true, a cls token is added in the beginning of each
                input sequence.
        """
        super().__init__()
        assert (
            len(patch_embed_shape) == 3
        ), "Patch_embed_shape should be in the form of (T, H, W)."
        self.cls_embed_on = has_cls
        self.sep_pos_embed = sep_pos_embed
        self._patch_embed_shape = patch_embed_shape
        self.num_spatial_patch = patch_embed_shape[1] * patch_embed_shape[2]
        self.num_temporal_patch = patch_embed_shape[0]

        if self.cls_embed_on:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            num_patches = self.num_spatial_patch * self.num_temporal_patch + 1
        else:
            num_patches = self.num_spatial_patch * self.num_temporal_patch

        if self.sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(1, self.num_spatial_patch, embed_dim)
            )
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, self.num_temporal_patch, embed_dim)
            )
            if self.cls_embed_on:
                self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

    @property
    def patch_embed_shape(self):
        return self._patch_embed_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor.
        """
        B, N, C = x.shape
        if self.cls_embed_on:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.repeat(
                1, self.num_temporal_patch, 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.num_spatial_patch,
                dim=1,
            )
            if self.cls_embed_on:
                pos_embed = torch.cat([self.pos_embed_class, pos_embed], 1)
            x = x + pos_embed
        else:
            x = x + self.pos_embed

        return x



class BeitRelativePositionBias(nn.Module):
    def __init__(self, config, window_size):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, config.num_attention_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH
        # cls to token & token 2 cls & cls to cls

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :] # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = torch.zeros(
            size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype
        )
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self):
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] + 1, self.window_size[0] * self.window_size[1] + 1, -1
        )  # Wh*Ww,Wh*Ww,nH

        return relative_position_bias.permute(2, 0, 1).contiguous()

class BeitRelativePositionBias3d(nn.Module):
    def __init__(self, config, window_size):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1)+ 3
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, config.num_attention_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH
        # cls to token & token 2 cls & cls to cls

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords_z = torch.arange(window_size[2])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w, coords_z]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = torch.zeros(
            size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype
        )
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self):
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] + 1, self.window_size[0] * self.window_size[1] + 1, -1
        )  # Wh*Ww,Wh*Ww,nH

        return relative_position_bias.permute(2, 0, 1).contiguous()



class PositionEmbed(nn.Module):
    def __init__(self, num_patches=196, d_model=768, num_tokens=0):
        super().__init__()

        # Compute the positional encodings once in log space.
        self.num_tokens = num_tokens
        assert self.num_tokens >=0, "num_tokens must be class token or no, so must 0 or 1"
        pe = torch.zeros(num_patches+self.num_tokens, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, num_patches + self.num_tokens).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        pe = pe.cuda()
        self.register_buffer('pe', pe)

    def __call__(self):
        return self.pe
