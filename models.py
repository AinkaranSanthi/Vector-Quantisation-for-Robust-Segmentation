import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from torch import einsum
import math
from layers import VectorQuantiser, Encoder, Decoder,  ResnetBlock, PositionalEncodingPermute3D, PositionalEncodingPermute2D, Transformer

class UNet(nn.Module):

   def __init__(self, channels,groups, in_ch, out_ch, enc_blocks, dec_blocks, dim, act= None,  with_conv = True, dropout=0.0):
        super().__init__()
        if len (enc_blocks) != len(dec_blocks) + 1:
           raise Exception('length of list of encoder blocks should be 1 greater than the list of decoder blocks')
        self.encoder = Encoder(channels=channels,groups=groups, blocks=enc_blocks, dim=dim, act= act,  dropout=dropout)
        self.dec_channels = channels * (2**len(enc_blocks))
        self.decoder = Decoder(channels=self.dec_channels, out_ch = out_ch, groups=groups, blocks=dec_blocks, dim=dim, act= act,  with_conv = with_conv, dropout=dropout)

   def forward(self, x):
        out_e = self.encoder(x)
        out = self.decoder(out_e)
        return out, out_e

class VQUNet(nn.Module):

    def __init__(self, *, channels,groups, in_ch, out_ch, enc_blocks, dec_blocks, dim, embed_dim, n_e,  image_size, quantise = 'spatial', act= None,  with_conv = True, VQ = True, pos_encoding = False, dropout=0.0):
        super().__init__()
        if len (enc_blocks) != len(dec_blocks) + 1:
           raise Exception('length of list of encoder blocks should be 1 greater than the list of decoder blocks')
        self.encoder = Encoder(channels=channels,in_ch = in_ch, groups=groups, blocks=enc_blocks, dim=dim, act= act,  dropout=dropout)
        self.dec_channels = channels * (2**len(enc_blocks))
        self.VQ = VQ
        self.quant_dim = embed_dim if quantise == 'spatial' else math.prod([image_size[i]//(2**len(enc_blocks)) for i in range(len(image_size))])
        self.pre_quant_conv = torch.nn.Conv2d(self.dec_channels, embed_dim, kernel_size=1,stride=1, padding=0) if dim == '2D' else torch.nn.Conv3d(self.dec_channels, embed_dim, kernel_size=1,stride=1, padding=0)
        self.p_enc = PositionalEncodingPermute2D(embed_dim) if dim == '2D' else PositionalEncodingPermute3D(embed_dim) if pos_encoding == True else nn.Identity()
        self.quantise = VectorQuantiser(n_e = n_e, e_dim = self.quant_dim, quantise = quantise, dim = dim, beta=0.25) if self.VQ == True else nn.Identity()
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, self.dec_channels, kernel_size=1,stride=1, padding=0) if dim == '2D' else torch.nn.Conv3d(embed_dim, self.dec_channels, kernel_size=1,stride=1, padding=0)
        self.midblock = ResnetBlock(in_channels=self.dec_channels, out_channels=self.dec_channels,dim = dim,  groups = groups, act = act, dropout = dropout)

        self.decoder = Decoder(channels=self.dec_channels, out_ch = out_ch,  groups=groups, blocks=dec_blocks, dim=dim, act= act,  with_conv = with_conv, dropout=dropout)

    def forward(self, x):
        out_e = self.encoder(x)
        q = self.pre_quant_conv(out_e[-1])
        q = self.p_enc(q)
        if self.VQ == True:
           vq, q_loss, info = self.quantise(q)
        else:
           vq, q_loss, info = self.quantise(q), 0, 0 
        vq_post = self.post_quant_conv(vq)
        vq_post = self.midblock(vq_post)
        dec_in = out_e[:-1]
        dec_in.append(vq_post)
        out = self.decoder(dec_in)
        return out, vq, q_loss, info


class transUNet(nn.Module):

   def __init__(self, trans_layers, image_size, hidden_dim, num_heads, channels,groups, in_ch, out_ch, enc_blocks, dec_blocks, dim, act= None,  with_conv = True, trans_attn='spatial', conv_dropout=0.0, trans_dropout=0.2):
        super().__init__()
        #Determine whether to perform attention channel or spatial wise
        if len (enc_blocks) != len(dec_blocks) + 1:
           raise Exception('length of list of encoder blocks should be 1 greater than the list of decoder blocks')
        self.trans_attn = trans_attn
        self.dim = dim
        self.encoder = Encoder(channels=channels,in_ch = in_ch,groups=groups, blocks=enc_blocks, dim=dim, act= act, dropout=conv_dropout)
        self.dec_channels = channels * (2**len(enc_blocks))
        self.trans_dim = math.prod([image_size[i]//(2**len(enc_blocks)) for i in range(len(image_size))])

        self.tblocks = nn.ModuleList(
                [TransformerBlock(self.dec_channels if self_trans_attn == 'spatial' else self.trans_dim, hidden_dim, num_heads, trans_dropout) for i in range(trans_layers)]
        )
        self.decoder = Decoder(channels=self.dec_channels, out_ch = out_ch, groups=groups, blocks=dec_blocks, dim=dim, act= act,  with_conv = with_conv, dropout=conv_dropout)

   def forward(self, x):
        out_e = self.encoder(x)
        if self.dim == '3D':
           out_t =  einops.rearrange(out_e[-1], "b c h w d -> b (h w d) c") if self.trans_attn == 'spatial' else einops.rearrange(x, "b c h w d -> b c (h w d)")
        else:
           out_t =  einops.rearrange(out_e[-1], "b c h w -> b (h w) c") if self.trans_attn == 'spatial' else einops.rearrange(x, "b c h w -> b c (h w)")
        for blk in self.tblocks:
           out_t = blk(out_t)
        if self.dim == '3D':
           out_t =  einops.rearrange(out_t, "b (h w d) c -> b c h w d", h = out_e[-1].shape[2], w = out_e[-1].shape[3], d = out_e[-1].shape[4]) if self.trans_attn == 'spatial' \
           else einops.rearrange(x, "b c h w d -> b c (h w d)", h = out_e[-1].shape[2], w = out_e[-1].shape[3], d = out_e[-1].shape[4])
        else:
           out_t =  einops.rearrange(out_t, "b (h w) c -> b c h w", h = out_e[-1].shape[2], w = out_e[-1].shape[3]) if self.trans_attn == 'spatial' \
           else einops.rearrange(x, "b c h w -> b c (h w )", h = out_e[-1].shape[2], w = out_e[-1].shape[3])
        dec_in = out_e[1:].insert(0, out_t)
        out = self.decoder(dec_in)
        return out, out_e

class VQtransUNet(nn.Module):

   def __init__(self, *, trans_layers, image_size, hidden_dim, num_heads, channels,groups, in_ch, out_ch, enc_blocks, dec_blocks, dim, act= None,  with_conv = True,
        embed_dim, n_e,  quantise = 'spatial',VQ = True, pos_encoding = False, trans_attn='spatial', conv_dropout=0.0, trans_dropout=0.2):
        super().__init__()
        #Determine whether to perform attention channel or spatial wise
        if len (enc_blocks) != len(dec_blocks) + 1:
           raise Exception('length of list of encoder blocks should be 1 greater than the list of decoder blocks')
        self.trans_attn = trans_attn
        self.dim = dim
        self.VQ = VQ
        self.encoder = Encoder(channels=channels,in_ch = in_ch,groups=groups, blocks=enc_blocks, dim=dim, act= act,  dropout=conv_dropout)
        self.dec_channels = channels * (2**len(enc_blocks))
        self.trans_dim = math.prod([image_size[i]//(2**len(enc_blocks)) for i in range(len(image_size))])
        self.quant_dim = embed_dim if quantise == 'spatial' else self.trans_dim
        self.pre_quant_conv = torch.nn.Conv2d(self.dec_channels, embed_dim, kernel_size=1,stride=1, padding=0) if dim == '2D' else torch.nn.Conv3d(self.dec_channels, embed_dim, kernel_size=1,stride=1, padding=0)
        self.p_enc = PositionalEncodingPermute2D(embed_dim) if dim == '2D' else PositionalEncodingPermute3D(embed_dim) if pos_encoding == True else nn.Identity()
        self.quantize = VectorQuantiser(n_e = n_e, e_dim = self.quant_dim, quantise = quantise, dim = dim, beta=0.25) if self.VQ == True else nn.Identity()
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, self.dec_channels, kernel_size=1,stride=1, padding=0) if dim == '2D' else torch.nn.Conv3d(embed_dim, self.dec_channels, kernel_size=1,stride=1, padding=0)
        self.tblocks = nn.ModuleList(
                [TransformerBlock(self.dec_channels if self_trans_attn == 'spatial' else self.trans_dim, hidden_dim, num_heads, trans_dropout) for i in range(trans_layers)]
        )
        self.decoder = Decoder(channels=self.dec_channels, out_ch = out_ch, groups=groups, blocks=dec_blocks, dim=dim, act= act,  with_conv = with_conv, dropout=conv_dropout)

   def forward(self, x):
        out_e = self.encoder(x)
        q = self.pre_quant_conv(out_e[-1])
        q = self.p_enc(q)
        if self.VQ == True:
           vq, q_loss, info = self.quantize(q) 
        else:
           vq, q_loss, info = self.quantize(q), 0, 0 
        vq_post = self.post_quant_conv(vq)
        if self.dim == '3D':
           out_t =  einops.rearrange(vq_post, "b c h w d -> b (h w d) c") if self.trans_attn == 'spatial' else einops.rearrange(x, "b c h w d -> b c (h w d)")
        else:
           out_t =  einops.rearrange(vq_post, "b c h w -> b (h w) c") if self.trans_attn == 'spatial' else einops.rearrange(x, "b c h w -> b c (h w)")
        for blk in self.tblocks:
           out_t = blk(out_t)
        if self.dim == '3D':
           out_t =  einops.rearrange(vq_post, "b (h w d) c -> b c h w d", h = out_e.shape[2], w = out_e.shape[3], d = out_e.shape[4]) if self.trans_attn == 'spatial' \
           else einops.rearrange(x, "b c h w d -> b c (h w d)", h = out_e.shape[2], w = out_e.shape[3], d = out_e.shape[4])
        else:
           out_t =  einops.rearrange(vq_post, "b (h w) c -> b c h w", h = out_e.shape[2], w = out_e.shape[3]) if self.trans_attn == 'spatial' \
           else einops.rearrange(x, "b c h w -> b c (h w )", h = out_e.shape[2], w = out_e.shape[3])
        dec_in = out_e[:-1]
        dec_in.append(out_t)
        out = self.decoder(dec_in)
        return out, vq, q_loss, info
