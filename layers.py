import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from torch import einsum
import math

def nonlinearity(x, act='swish'):
    #swish actication if act is swish else pick activation of your choice
    if act == 'swish':
       nl = x*torch.sigmoid(x)
    else:
       act = eval(act) 
       nl = act(x)

    return nl


def Normalise(num_groups, in_channels, dim):
    #batch normalisation if num_groups is none otherwise group normalisation
    if num_groups ==None:
        return torch.nn.BatchNorm3d(in_channels) if dim =='3D' else torch.nn.BatchNorm2d(in_channels)
    else:
        assert in_channels % num_groups == 0
        return torch.nn.GroupNorm(num_groups, in_channels)

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels,out_channels,groups,dim,act= None,
                 dropout=0.0):
        super().__init__()
        # Pre-activation convolutional residual block for 2D or 3D input; "dim"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act

        self.norm1 = Normalise(in_channels = in_channels, num_groups = groups, dim = dim)
        self.conv1 = torch.nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1) if dim == '2D' else torch.nn.Conv3d(in_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.norm2 = Normalise(in_channels =  out_channels, num_groups = groups, dim = dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1) if dim == '2D' else torch.nn.Conv3d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
        
        if self.in_channels != self.out_channels:
           self.conv_skip = torch.nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0) if dim == '2D' else torch.nn.Conv3d(in_channels,out_channels,kernel_size=1,stride=1,padding=0)

    def forward(self, x):
        h = self.norm1(x)
        h = nonlinearity(h, act=self.act)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h, act = self.act)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
           x = self.conv_skip(x)

        return x+h


class Upsample(nn.Module):
    def __init__(self, in_channels, dim, with_conv):
        super().__init__()
        # Upsampling block for 2D or 3D input; "dim" using linear interpolation with or without a convolutional layer; "with_conv"
        self.with_conv = with_conv
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if dim == '2D' else nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1) if dim == '2D' else torch.nn.Conv3d(in_channels,out_channels,kernel_size=3,stride=1,padding=1)


    def forward(self, x):
        x = self.up(x)
        if self.with_conv:
            x = self.conv(x)
        return x    

class Encoder(nn.Module):
    """
      This class creates an encoder with any number of "levels" with each level
      with each level consisting of any number of pre activation residual convolutional  "blocks"
      for handling 2D or 3D dimension ("dim") input
      """

    def __init__(self, in_ch, channels,groups, blocks, dim, act= None,  dropout=0.0):
        super().__init__()
        self.enc = nn.ModuleList()
        levels = len(blocks)
        for i in range(levels):
            block = []
            for j in range(blocks[i]):
                out_channels = channels *2 if j == 0 else channels                        
                if i == 0:
                   block.append(torch.nn.Conv2d(in_ch, channels, kernel_size = 3, stride = 1, padding = 1) if dim == '2D' else torch.nn.Conv3d(in_ch, channels, kernel_size = 3, stride = 1, padding = 1))
                block.append(ResnetBlock(in_channels = channels, out_channels = out_channels,dim = dim,  groups = groups, act = act, dropout = dropout))          
                channels = out_channels
            self.enc.append(nn.Sequential(*block))
            if i != levels:
               self.enc.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=2) if dim == '2D' else nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, stride=2))

    def forward(self, x):
        outs = []
        for l, level in enumerate(self.enc):
            x = level(x)
            if l % 2 ==0:
               outs.append(x)
        return outs

class Decoder(nn.Module):
      """
      This class creates a decoder with any number of "levels" with each level
      with each level consisting of any number of pre activation residual convolutional  "blocks" 
      for handling 2D or 3D dimension ("dim") input
      """
      def __init__(self, channels,groups, out_ch, blocks, dim, act= None,  with_conv = True, dropout=0.0):
        super().__init__()
        levels = len(blocks) 
        self.dec = nn.ModuleList()
        for i in range(levels):
            block = []
            self.dec.append(Upsample(in_channels = channels, dim=dim,  with_conv = with_conv))
            for j in range(blocks[i]):
                channels = channels + channels//2
                if j == 0:
                   out_channels = (channels//3)
                else: 
                   out_channels = channels
                block.append(ResnetBlock(in_channels = channels, out_channels = out_channels, dim = dim, groups = groups, act = act, dropout = dropout))
                channels = out_channels
                if i == levels-1:
                   block.append(torch.nn.Conv2d(channels, out_ch, kernel_size = 3, stride = 1, padding = 1) if dim == '2D' else torch.nn.Conv3d(channels, out_ch, kernel_size = 3, stride = 1, padding = 1)) 
            self.dec.append(nn.Sequential(*block))
      def forward(self, x):
        x1 = x[-1]
        enc = list(reversed(x))
        for j, level in enumerate(self.dec):
            if j % 2 == 0:
               x1 = level(x1)
            else:
               x1 = level(torch.cat((enc[(j+1)//2], x1), dim = 1))
        return x1


class VectorQuantiser(nn.Module):
    """
    This class is adapted from https://github.com/CompVis/taming-transformers
    https://arxiv.org/pdf/2012.09841.pdf
    """
    def __init__(self, n_e, e_dim, beta = 0.2, dim = '3D', quantise = 'spatial', legacy=False):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.dim = dim
        self.legacy = legacy
        self.quantise = quantise
        

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        self.ed = lambda x: [torch.norm(x[i]) for i in range(x.shape[0])]

    def dist(self, u, v):
         d = torch.sum(u ** 2, dim=1, keepdim=True) + \
             torch.sum(v**2, dim=1) - 2 * \
             torch.einsum('bd,dn->bn', u, rearrange(v, 'n d -> d n'))

         return d
    
    def geodist(self, u, v):
        d1 = torch.einsum('bd,dn->bn', self.embedding.weight, rearrange(self.embedding.weight, 'n d -> d n'))
        ed1 = torch.tensor(self.ed(self.embedding.weight)).repeat(self.n_e, 1)
        ed2 = ed1.transpose(0,1)
        geod = torch.clamp(d1/(ed1*ed2), min=-0.99999, max=0.99999)
        
        return torch.acos(geod)
 

    def forward(self, z):
        #Determine to quantise either spatial or channel wise
        if self.quantise == 'spatial':
           # reshape z  and flatten
           if self.dim == '2D':
               z = rearrange(z, 'b c h w -> b h w c').contiguous()
           else:    
               z = rearrange(z, 'b c h w z -> b h w z c').contiguous()
        z_flattened = z.view(-1, self.e_dim)
        
        # compute distances from z to codebook
        d = self.dist(z_flattened, self.embedding.weight)
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        
        # compute mean codebook distances
        cd = self.dist(self.embedding.weight, self.embedding.weight)
        min_distance = torch.kthvalue(cd, 2, 0)
        mean_cb_distance = torch.mean(min_distance[0])
            
        # compute mean codebook variance
        mean_cb_variance = torch.mean(torch.var(cd, 1))

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()
        
        # reshape to original input shape
        if self.quantise == 'spatial':
           if self.dim == '2D':
              z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
           else:
              z_q = rearrange(z_q, 'b h w z c -> b c h w z').contiguous()
        
        # Get Sampled Indices
        min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3], z_q.shape[4])
        sampled_idx = torch.zeros(z.shape[0]*self.n_e).to(z.device)
        sampled_idx[min_encoding_indices] = 1
        sampled_idx = sampled_idx.view(z.shape[0], self.n_e)
        return z_q, loss, (min_encoding_indices, sampled_idx, mean_cb_distance, mean_cb_variance)

class PositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        """
        The code for this class is forked from https://github.com/tatp22/multidim-positional-encoding
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding3D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")
        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        pos_z = torch.arange(z, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = (
            torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
                .unsqueeze(1)
                .unsqueeze(1)
        )
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1).unsqueeze(1)
        emb_z = torch.cat((sin_inp_z.sin(), sin_inp_z.cos()), dim=-1)
        emb = torch.zeros((x, y, z, self.channels * 3), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, :, : self.channels] = emb_x
        emb[:, :, :, self.channels: 2 * self.channels] = emb_y
        emb[:, :, :, 2 * self.channels:] = emb_z

        return emb[None, :, :, :, :orig_ch].repeat(batch_size, 1, 1, 1, 1)

class PositionalEncodingPermute3D(nn.Module):
    def __init__(self, channels):
        """
        The code for this class is forked from https://github.com/tatp22/multidim-positional-encoding
        Accepts (batchsize, ch, x, y, z) instead of (batchsize, x, y, z, ch)
        """
        super(PositionalEncodingPermute3D, self).__init__()
        self.penc = PositionalEncoding3D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 4, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 4, 1, 2, 3)

    @property
    def org_channels(self):
        return self.penc.org_channels


class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        The code for this class is forked from https://github.com/tatp22/multidim-positional-encoding
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb = torch.zeros((x, y, self.channels * 2), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels : 2 * self.channels] = emb_y

        return emb[None, :, :, :orig_ch].repeat(tensor.shape[0], 1, 1, 1)


class PositionalEncodingPermute2D(nn.Module):
    def __init__(self, channels):
        """
        The code for this class is forked from https://github.com/tatp22/multidim-positional-encoding
        Accepts (batchsize, ch, x, y) instead of (batchsize, x, y, ch)
        """
        super(PositionalEncodingPermute2D, self).__init__()
        self.penc = PositionalEncoding2D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 3, 1, 2)

    @property
    def org_channels(self):
        return self.penc.org_channels


class MLP(nn.Module):
    """
    A multi-layer perceptron block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(self, dim_size, hidden_dim, dropout_rate):

        super().__init__()


        self.l1 = nn.Linear(dim_size, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, dim_size)
        self.act = nn.GELU()
        self.d1 = nn.Dropout(dropout_rate)
        self.d2 = nn.Dropout(dropout_rate)


    def forward(self, x):
        x = self.act(self.l1(x))
        x = self.d1(x)
        x = self.l2(x)
        x = self.d2(x)
        return x

class MHSA(nn.Module):
    """
    A self-attention block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(self, dim_size, num_heads, dropout_rate):
        super().__init__()


        if hidden_dim % num_heads != 0:
            raise ValueError("hidden dim must be divisible by num_heads.")

        self.num_heads = num_heads
        self.out = nn.Linear(dim_size, dim_size)
        self.qkv = nn.Linear(dim_size, dim_size * 3, bias=False)
        self.drop_out = nn.Dropout(dropout_rate)
        self.drop_attn = nn.Dropout(dropout_rate)
        self.head_dim = dim_size // num_heads
        self.scale = self.head_dim**-0.5
        self.out = nn.Linear(dim_size, dim_size)

    def forward(self, x):
        q, k, v = einops.rearrange(self.qkv(x), "b c (qkv h d) -> qkv b h c d", qkv=3, h=self.num_heads)
        attn_m = (torch.einsum("bhxd,bhyd->bhxy", q, k) * self.scale).softmax(dim=-1)
        attn_m = self.drop_attn(attn_m)
        x = torch.einsum("bhxy,bhzd->bhxd", attn_m, v)
        x = einops.rearrange(x, "b h c d -> b c (h d)")
        x = self.out(x)
        x = self.drop_out(x)
        return x

class Transformer(nn.Module):
    """
    A transformer block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(self, dim_size, hidden_dim, num_heads, dropout_rate):

        super().__init__()

        self.mlp = MLP(dim_size, hidden_dim, dropout_rate)
        self.n1 = nn.LayerNorm(dim_size)
        self.attn = MHSA(dim_size, num_heads, dropout_rate)
        self.n2 = nn.LayerNorm(dim_size)


    def forward(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x


