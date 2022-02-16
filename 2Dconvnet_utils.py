import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from torch import einsum

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=8, num_channels=in_channels, eps=1e-6, affine=True)

class ResnetBlock2D(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class VectorQuantizer2D(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, n_e, e_dim, beta, remap=None, unknown_index="random",
                 sane_index_shape=False, legacy=True):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0],-1) # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0],-1) # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1) # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q

class GumbelQuantize2D(nn.Module):
    """
    credit to @karpathy: https://github.com/karpathy/deep-vector-quantization/blob/main/model.py (thanks!)
    Gumbel Softmax trick quantizer
    Categorical Reparameterization with Gumbel-Softmax, Jang et al. 2016
    https://arxiv.org/abs/1611.01144
    """
    def __init__(self, embedding_dim, n_embed, straight_through=True,
                 kl_weight=5e-4, temp_init=1.0, use_vqinterface=True,
                 remap=None, unknown_index="random"):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_embed = n_embed

        self.straight_through = straight_through
        self.temperature = temp_init
        self.kl_weight = kl_weight

        self.proj = nn.Conv2d(embedding_dim, n_embed, 1)
        self.embed = nn.Embedding(n_embed, embedding_dim)

        self.use_vqinterface = use_vqinterface

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_embed} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_embed

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, return_logits=False):
        # force hard = True when we are in eval mode, as we must quantize. actually, always true seems to work
        hard = self.straight_through if self.training else True
        temp = self.temperature if temp is None else temp

        logits = self.proj(z)
        if self.remap is not None:
            # continue only with used logits
            full_zeros = torch.zeros_like(logits)
            logits = logits[:,self.used,...]

        soft_one_hot = F.gumbel_softmax(logits, tau=temp, dim=1, hard=hard)
        if self.remap is not None:
            # go back to all entries but unused set to zero
            full_zeros[:,self.used,...] = soft_one_hot
            soft_one_hot = full_zeros
        z_q = einsum('b n h w, n d -> b d h w', soft_one_hot, self.embed.weight)

        # + kl divergence to the prior loss
        qy = F.softmax(logits, dim=1)
        diff = self.kl_weight * torch.sum(qy * torch.log(qy * self.n_embed + 1e-10), dim=1).mean()

        ind = soft_one_hot.argmax(dim=1)
        if self.remap is not None:
            ind = self.remap_to_used(ind)
        if self.use_vqinterface:
            if return_logits:
                return z_q, diff, (None, None, ind), logits
            return z_q, diff, (None, None, ind)
        return z_q, diff, ind

    def get_codebook_entry(self, indices, shape):
        b, h, w, c = shape
        assert b*h*w == indices.shape[0]
        indices = rearrange(indices, '(b h w) -> b h w', b=b, h=h, w=w)
        if self.remap is not None:
            indices = self.unmap_to_all(indices)
        one_hot = F.one_hot(indices, num_classes=self.n_embed).permute(0, 3, 1, 2).float()
        z_q = einsum('b n h w, n d -> b d h w', one_hot, self.embed.weight)
        return z_q


class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
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

class Upsample2D(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)


    def forward(self, x):
        x = self.up(x)
        if self.with_conv:
            x = self.conv(x)
        return x

class GumbelUNet2Dpos(nn.Module):

    def __init__(self, num_classes, inputchannels = 1, channels = 32, dropout = 0.0, n_embed = 1024,
                 embed_dim = 512, kl_weight=1e-8, remap=None,):
        super(GumbelUNet2Dpos, self).__init__()
        use_bias = True
        self.conv11 = nn.Conv2d(inputchannels, channels, kernel_size=3, padding=1, stride=1, bias=use_bias)
        self.conv12 = ResnetBlock2D(in_channels=channels, out_channels=channels,  dropout=dropout)
        self.down1 = nn.Conv2d(channels, channels*2, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv21 = ResnetBlock2D(in_channels=channels*2, out_channels=channels*2,  dropout=dropout)
        self.down2 = nn.Conv2d(channels*2, channels*4, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv31 = ResnetBlock2D(in_channels=channels*4, out_channels=channels*4,  dropout=dropout)
        self.down3 = nn.Conv2d(channels*4, channels*8, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv41 = ResnetBlock2D(in_channels=channels*8, out_channels=channels*8, dropout=dropout)
        self.down4 = nn.Conv2d(channels*8, channels*16, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv51 = ResnetBlock2D(in_channels=channels*16, out_channels=channels*16, dropout=dropout)
        self.quant_conv = torch.nn.Conv2d(channels*16, embed_dim, 1)
        self.p_enc_3d = PositionalEncodingPermute2D(channels*16)
        self.quantize = GumbelQuantize2D(embedding_dim=embed_dim,
                                       n_embed=n_embed,
                                       kl_weight=kl_weight, temp_init=1.0,
                                       remap=remap)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, channels*16, 1)
        self.conv52 = ResnetBlock2D(in_channels=channels*16, out_channels=channels*16,dropout=dropout)
        self.up4 =  Upsample2D(channels*16, True)
        self.conv42 = ResnetBlock2D(in_channels=channels*24, out_channels=channels*8, dropout=dropout)
        self.up3 =  Upsample2D(channels*8, True)
        self.conv32 = ResnetBlock2D(in_channels=channels*12, out_channels=channels*4, dropout=dropout)
        self.up2 = Upsample2D(channels*4, True)
        self.conv22 = ResnetBlock2D(in_channels=channels*6, out_channels=channels*2, dropout=dropout)
        self.up1 = Upsample2D(channels*2, True)
        self.conv13 = ResnetBlock2D(in_channels=channels*3, out_channels=channels, dropout=dropout)
        self.conv14 = nn.Conv2d(channels, num_classes, kernel_size=1, padding=0, bias=use_bias)

    def forward(self, x):
        x1 = self.conv11(x)
        x1 = self.conv12(x1)
        x2 = self.down1(x1)
        x2 = self.conv21(x2)
        x3 = self.down2(x2)
        x3 = self.conv31(x3)
        x4 = self.down3(x3)
        x4 = self.conv41(x4)
        x5 = self.down4(x4)
        x5 = self.conv51(x5)
        x5 = self.quant_conv(x5)
        x5 = self.p_enc_3d(x5)
        quant, emb_loss, info = self.quantize(x5)
        x5 = self.post_quant_conv(quant)
        x5 = self.conv52(x5)
        x4 = torch.cat([self.up4(x5), x4], dim=1)
        x4 = self.conv42(x4)
        x3 = torch.cat([self.up3(x4), x3], dim=1)
        x3 = self.conv32(x3)
        x2 = torch.cat([self.up2(x3), x2], dim=1)
        x2 = self.conv22(x2)
        x1 = torch.cat([self.up1(x2), x1], dim=1)
        x1 = self.conv13(x1)
        x = self.conv14(x1)




        return x, emb_loss, quant

class UNet2D(nn.Module):

    def __init__(self, num_classes, inputchannels = 1, channels = 32, dropout = 0.0):
        super(UNet2D, self).__init__()
        use_bias = True
        self.conv11 = nn.Conv2d(inputchannels, channels, kernel_size=3, padding=1, stride=1, bias=use_bias)
        self.conv12 = ResnetBlock2D(in_channels=channels, out_channels=channels,  dropout=dropout)
        self.down1 = nn.Conv2d(channels, channels*2, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv21 = ResnetBlock2D(in_channels=channels*2, out_channels=channels*2,  dropout=dropout)
        self.down2 = nn.Conv2d(channels*2, channels*4, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv31 = ResnetBlock2D(in_channels=channels*4, out_channels=channels*4,  dropout=dropout)
        self.down3 = nn.Conv2d(channels*4, channels*8, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv41 = ResnetBlock2D(in_channels=channels*8, out_channels=channels*8, dropout=dropout)
        self.down4 = nn.Conv2d(channels*8, channels*16, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv51 = ResnetBlock2D(in_channels=channels*16, out_channels=channels*16, dropout=dropout)

        self.conv52 = ResnetBlock2D(in_channels=channels*16, out_channels=channels*16,dropout=dropout)
        self.up4 =  Upsample2D(channels*16, True)
        self.conv42 = ResnetBlock2D(in_channels=channels*24, out_channels=channels*8, dropout=dropout)
        self.up3 =  Upsample2D(channels*8, True)
        self.conv32 = ResnetBlock2D(in_channels=channels*12, out_channels=channels*4, dropout=dropout)
        self.up2 = Upsample2D(channels*4, True)
        self.conv22 = ResnetBlock2D(in_channels=channels*6, out_channels=channels*2, dropout=dropout)
        self.up1 = Upsample2D(channels*2, True)
        self.conv13 = ResnetBlock2D(in_channels=channels*3, out_channels=channels, dropout=dropout)
        self.conv14 = nn.Conv2d(channels, num_classes, kernel_size=1, padding=0, bias=use_bias)

    def forward(self, x):
        x1 = self.conv11(x)
        x1 = self.conv12(x1)
        x2 = self.down1(x1)
        x2 = self.conv21(x2)
        x3 = self.down2(x2)
        x3 = self.conv31(x3)
        x4 = self.down3(x3)
        x4 = self.conv41(x4)
        x5 = self.down4(x4)
        x5 = self.conv51(x5)

        x5 = self.conv52(x5)
        x4 = torch.cat([self.up4(x5), x4], dim=1)
        x4 = self.conv42(x4)
        x3 = torch.cat([self.up3(x4), x3], dim=1)
        x3 = self.conv32(x3)
        x2 = torch.cat([self.up2(x3), x2], dim=1)
        x2 = self.conv22(x2)
        x1 = torch.cat([self.up1(x2), x1], dim=1)
        x1 = self.conv13(x1)
        x = self.conv14(x1)

        return x, x5



class VQUNet2Dpos(nn.Module):

    def __init__(self, num_classes, inputchannels = 1, channels = 32, dropout = 0.0, n_embed = 1024,
                 embed_dim = 512):
        super(VQUNet2Dpos, self).__init__()
        use_bias = True
        self.conv11 = nn.Conv2d(inputchannels, channels, kernel_size=3, padding=1, stride=1, bias=use_bias)
        self.conv12 = ResnetBlock2D(in_channels=channels, out_channels=channels,  dropout=dropout)
        self.down1 = nn.Conv2d(channels, channels*2, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv21 = ResnetBlock2D(in_channels=channels*2, out_channels=channels*2,  dropout=dropout)
        self.down2 = nn.Conv2d(channels*2, channels*4, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv31 = ResnetBlock2D(in_channels=channels*4, out_channels=channels*4,  dropout=dropout)
        self.down3 = nn.Conv2d(channels*4, channels*8, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv41 = ResnetBlock2D(in_channels=channels*8, out_channels=channels*8, dropout=dropout)
        self.down4 = nn.Conv2d(channels*8, channels*16, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv51 = ResnetBlock2D(in_channels=channels*16, out_channels=channels*16, dropout=dropout)
        self.quant_conv = torch.nn.Conv2d(channels*16, embed_dim, 1)
        self.p_enc_3d = PositionalEncodingPermute2D(channels*16)
        self.quantize = VectorQuantizer2D(n_embed, channels*16, beta=0.25)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, channels*16, 1)
        self.conv52 = ResnetBlock2D(in_channels=channels*16, out_channels=channels*16,dropout=dropout)
        self.up4 =  Upsample2D(channels*16, True)
        self.conv42 = ResnetBlock2D(in_channels=channels*24, out_channels=channels*8, dropout=dropout)
        self.up3 =  Upsample2D(channels*8, True)
        self.conv32 = ResnetBlock2D(in_channels=channels*12, out_channels=channels*4, dropout=dropout)
        self.up2 = Upsample2D(channels*4, True)
        self.conv22 = ResnetBlock2D(in_channels=channels*6, out_channels=channels*2, dropout=dropout)
        self.up1 = Upsample2D(channels*2, True)
        self.conv13 = ResnetBlock2D(in_channels=channels*3, out_channels=channels, dropout=dropout)
        self.conv14 = nn.Conv2d(channels, num_classes, kernel_size=1, padding=0, bias=use_bias)

    def forward(self, x):
        x1 = self.conv11(x)
        x1 = self.conv12(x1)
        x2 = self.down1(x1)
        x2 = self.conv21(x2)
        x3 = self.down2(x2)
        x3 = self.conv31(x3)
        x4 = self.down3(x3)
        x4 = self.conv41(x4)
        x5 = self.down4(x4)
        x5 = self.conv51(x5)
        x5 = self.quant_conv(x5)
        x5 = self.p_enc_3d(x5)
        quant, emb_loss, info = self.quantize(x5)
        x5 = self.post_quant_conv(quant)
        x5 = self.conv52(x5)
        x4 = torch.cat([self.up4(x5), x4], dim=1)
        x4 = self.conv42(x4)
        x3 = torch.cat([self.up3(x4), x3], dim=1)
        x3 = self.conv32(x3)
        x2 = torch.cat([self.up2(x3), x2], dim=1)
        x2 = self.conv22(x2)
        x1 = torch.cat([self.up1(x2), x1], dim=1)
        x1 = self.conv13(x1)
        x = self.conv14(x1)




        return x, emb_loss, quant
