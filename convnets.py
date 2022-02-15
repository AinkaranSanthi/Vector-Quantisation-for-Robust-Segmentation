import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=8, num_channels=in_channels, eps=1e-6, affine=True)

class VectorQuantizer2(nn.Module):
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
        z = rearrange(z, 'b c h w z -> b h w z c').contiguous()
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
        z_q = rearrange(z_q, 'b h w z c -> b c h w z').contiguous()

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
        
class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv3d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv3d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv3d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv3d(in_channels,
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


class VQUNet3D(nn.Module):

    def __init__(self, num_classes, inputchannels = 1, channels = 16, dropout = 0.0, n_embed = 1024,
                 embed_dim = 256):
        super(VQUNet3D, self).__init__()
        use_bias = True
        self.conv11 = nn.Conv3d(inputchannels, channels, kernel_size=3, padding=1, stride=1, bias=use_bias)
        self.conv12 = ResnetBlock(in_channels=channels, out_channels=channels,  dropout=dropout)
        self.down1 = nn.Conv3d(channels, channels*2, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv21 = ResnetBlock(in_channels=channels*2, out_channels=channels*2,  dropout=dropout)
        self.down2 = nn.Conv3d(channels*2, channels*4, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv31 = ResnetBlock(in_channels=channels*4, out_channels=channels*4,  dropout=dropout)
        self.down3 = nn.Conv3d(channels*4, channels*8, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv41 = ResnetBlock(in_channels=channels*8, out_channels=channels*8, dropout=dropout)
        self.down4 = nn.Conv3d(channels*8, channels*16, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv51 = ResnetBlock(in_channels=channels*16, out_channels=channels*16, dropout=dropout)
        self.quant_conv = torch.nn.Conv3d(channels*16, embed_dim, 1)
        self.quantize = VectorQuantizer2(n_embed, channels*16, beta=0.25)
        self.post_quant_conv = torch.nn.Conv3d(embed_dim, channels*16, 1)
        self.conv52 = ResnetBlock(in_channels=channels*16, out_channels=channels*16,dropout=dropout)
        self.up4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv42 = ResnetBlock(in_channels=channels*24, out_channels=channels*8, dropout=dropout)
        self.up3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv32 = ResnetBlock(in_channels=channels*12, out_channels=channels*4, dropout=dropout)
        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv22 = ResnetBlock(in_channels=channels*6, out_channels=channels*2, dropout=dropout)
        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv13 = ResnetBlock(in_channels=channels*3, out_channels=channels, dropout=dropout)
        self.conv14 = nn.Conv3d(channels, num_classes, kernel_size=1, padding=0, bias=use_bias)

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


class UNet3D(nn.Module):

    def __init__(self, num_classes, inputchannels = 1, channels = 16, dropout = 0.0, n_embed = 1024,
                 embed_dim = 256):
        super(UNet3D, self).__init__()
        use_bias = True
        self.conv11 = nn.Conv3d(inputchannels, channels, kernel_size=3, padding=1, stride=1, bias=use_bias)
        self.conv12 = ResnetBlock(in_channels=channels, out_channels=channels,  dropout=dropout)
        self.down1 = nn.Conv3d(channels, channels*2, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv21 = ResnetBlock(in_channels=channels*2, out_channels=channels*2,  dropout=dropout)
        self.down2 = nn.Conv3d(channels*2, channels*4, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv31 = ResnetBlock(in_channels=channels*4, out_channels=channels*4,  dropout=dropout)
        self.down3 = nn.Conv3d(channels*4, channels*8, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv41 = ResnetBlock(in_channels=channels*8, out_channels=channels*8, dropout=dropout)
        self.down4 = nn.Conv3d(channels*8, channels*16, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv5 = ResnetBlock(in_channels=channels*16, out_channels=channels*16, dropout=dropout)

        self.up4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv42 = ResnetBlock(in_channels=channels*24, out_channels=channels*8, dropout=dropout)
        self.up3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv32 = ResnetBlock(in_channels=channels*12, out_channels=channels*4, dropout=dropout)
        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv22 = ResnetBlock(in_channels=channels*6, out_channels=channels*2, dropout=dropout)
        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv13 = ResnetBlock(in_channels=channels*3, out_channels=channels, dropout=dropout)
        self.conv14 = nn.Conv3d(channels, num_classes, kernel_size=1, padding=0, bias=use_bias)

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
        x5 = self.conv5(x5)

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


class NLayerDiscriminator3DITN(nn.Module):
        """Defines a PatchGAN discriminator as in Pix2Pix
            --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
        """

        def __init__(self, input_size, input_nc=3, ndf=64, n_layers=3):
            """Construct a PatchGAN discriminator
            Parameters:
                input_nc (int)  -- the number of channels in input images
                ndf (int)       -- the number of filters in the last conv layer
                n_layers (int)  -- the number of conv layers in the discriminator
                norm_layer      -- normalization layer
            """
            super(NLayerDiscriminator3DITN, self).__init__()

            norm_layer = nn.BatchNorm3d
            num_features = torch.prod(torch.floor_divide(torch.tensor(input_size), 16))

            if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
                use_bias = norm_layer.func != nn.BatchNorm3d
            else:
                use_bias = norm_layer != nn.BatchNorm3d

            kw = 4
            padw = 1
            sequence = [nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
            nf_mult = 1
            nf_mult_prev = 1
            for n in range(1, n_layers):  # gradually increase the number of filters
                nf_mult_prev = nf_mult
                nf_mult = min(2 ** n, 8)
                sequence += [
                    nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                ]

            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n_layers, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

            sequence += [
                nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
            self.main = nn.Sequential(*sequence)
            down = [
                nn.Conv3d(ndf * nf_mult, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                nn.LeakyReLU(0.2, True)
            ]
            self.down = nn.Sequential(*down)

            self.fc = nn.Linear(ndf * nf_mult * num_features, ndf * nf_mult)
            self.translation = nn.Linear(ndf * nf_mult, 3)
            self.rotation = nn.Linear(ndf * nf_mult, 3)
            self.scaling = nn.Linear(ndf * nf_mult, 3)
            # self.shearing = nn.Linear(32, 3).to(self.device)

            self.translation.weight.data.zero_()
            self.translation.bias.data.copy_(torch.tensor([0, 0, 0], dtype=self.dtype))
            self.rotation.weight.data.zero_()
            self.rotation.bias.data.copy_(torch.tensor([0, 0, 0], dtype=self.dtype))
            self.scaling.weight.data.zero_()
            self.scaling.bias.data.copy_(torch.tensor([0, 0, 0], dtype=self.dtype))


        def forward(self, input):

            x = self.main(input)
            x5 = self.down(x)
            xa = F.avg_pool3d(x5, 2)
            xa = xa.view(xa.size(0), -1)
            xa = F.relu(self.fc(xa))
            self.affine_matrix(xa)

            b, c, d, h, w = x.size()
            id_grid = self.grid.unsqueeze(0).repeat(b, 1, 1, 1, 1)
            id_grid = id_grid.view(b, 3, -1)

            ones = torch.ones([b, 1, id_grid.size(2)]).to(self.device)

            self.T = torch.bmm(self.theta[:, 0:3, :], torch.cat((id_grid, ones), dim=1))
            self.T = self.T.view(b, 3, d, h, w)
            self.T = self.move_grid_dims(self.T)

            """Standard forward."""
            return x, self.warp_image

        def get_normalized_grid(self, size):
            ndim = len(size)
            grid = torch.stack(torch.meshgrid([torch.arange(s) for s in size]), 0).float()
            for i in range(ndim):
                grid[i] = 2. * grid[i] / (size[i] - 1) - 1.

            return grid

        def regularizer(self):
            return torch.tensor([0], dtype=self.dtype).to(self.device)

        def get_T(self):
            return self.T

        def get_T_inv(self):
            return self.T_inv

        def moveaxis(self, x, src, dst):
            ndims = x.dim()
            dims = list(range(ndims))
            dims.pop(src)
            if dst < 0:
                dst = ndims + dst
            dims.insert(dst, src)

            return x.permute(dims)

        def move_grid_dims(self, grid):
            size = grid.shape[2:]
            ndim = len(size)
            assert ndim == grid.shape[1]

            grid = self.moveaxis(grid, 1, -1)  # [batch, z?, y, x, dims]
            grid = grid[..., list(range(ndim))[::-1]]  # reverse dims

            return grid

        def transform(self, src, grid, interpolation='bilinear', padding='border'):
            return F.grid_sample(src, grid, mode=interpolation, align_corners=True, padding_mode=padding)

        def affine_matrix(self, x):
            b = x.size(0)

            trans = torch.tanh(self.translation(x)) * 1.0
            translation_matrix = torch.zeros([b, 4, 4], dtype=torch.float).to(self.device)
            translation_matrix[:, 0, 0] = 1.0
            translation_matrix[:, 1, 1] = 1.0
            translation_matrix[:, 2, 2] = 1.0
            translation_matrix[:, 0, 3] = trans[:, 0].view(-1)
            translation_matrix[:, 1, 3] = trans[:, 1].view(-1)
            translation_matrix[:, 2, 3] = trans[:, 2].view(-1)
            translation_matrix[:, 3, 3] = 1.0

            rot = torch.tanh(self.rotation(x)) * (math.pi / 4.0)
            # rot Z
            angle_1 = rot[:, 0].view(-1)
            rotation_matrix_1 = torch.zeros([b, 4, 4], dtype=torch.float).to(self.device)
            rotation_matrix_1[:, 0, 0] = torch.cos(angle_1)
            rotation_matrix_1[:, 0, 1] = -torch.sin(angle_1)
            rotation_matrix_1[:, 1, 0] = torch.sin(angle_1)
            rotation_matrix_1[:, 1, 1] = torch.cos(angle_1)
            rotation_matrix_1[:, 2, 2] = 1.0
            rotation_matrix_1[:, 3, 3] = 1.0
            # rot X
            angle_2 = rot[:, 1].view(-1)
            rotation_matrix_2 = torch.zeros([b, 4, 4], dtype=torch.float).to(self.device)
            rotation_matrix_2[:, 1, 1] = torch.cos(angle_2)
            rotation_matrix_2[:, 1, 2] = -torch.sin(angle_2)
            rotation_matrix_2[:, 2, 1] = torch.sin(angle_2)
            rotation_matrix_2[:, 2, 2] = torch.cos(angle_2)
            rotation_matrix_2[:, 0, 0] = 1.0
            rotation_matrix_2[:, 3, 3] = 1.0
            # rot Z
            angle_3 = rot[:, 2].view(-1)
            rotation_matrix_3 = torch.zeros([b, 4, 4], dtype=torch.float).to(self.device)
            rotation_matrix_3[:, 0, 0] = torch.cos(angle_3)
            rotation_matrix_3[:, 0, 1] = -torch.sin(angle_3)
            rotation_matrix_3[:, 1, 0] = torch.sin(angle_3)
            rotation_matrix_3[:, 1, 1] = torch.cos(angle_3)
            rotation_matrix_3[:, 2, 2] = 1.0
            rotation_matrix_3[:, 3, 3] = 1.0

            rotation_matrix = torch.bmm(rotation_matrix_1, rotation_matrix_2)
            rotation_matrix = torch.bmm(rotation_matrix, rotation_matrix_3)

            scale = torch.tanh(self.scaling(x)) * 1.0
            scaling_matrix = torch.zeros([b, 4, 4], dtype=torch.float).to(self.device)
            scaling_matrix[:, 0, 0] = torch.exp(scale[:, 0].view(-1))
            scaling_matrix[:, 1, 1] = torch.exp(scale[:, 1].view(-1))
            scaling_matrix[:, 2, 2] = torch.exp(scale[:, 2].view(-1))
            scaling_matrix[:, 3, 3] = 1.0

            # shear = torch.tanh(self.shearing(x)) * (math.pi / 4.0)
            #
            # shear_1 = shear[:, 0].view(-1)
            # shearing_matrix_1 = torch.zeros([b, 4, 4], dtype=torch.float).to(self.device)
            # shearing_matrix_1[:, 1, 1] = torch.cos(shear_1)
            # shearing_matrix_1[:, 1, 2] = -torch.sin(shear_1)
            # shearing_matrix_1[:, 2, 1] = torch.sin(shear_1)
            # shearing_matrix_1[:, 2, 2] = torch.cos(shear_1)
            # shearing_matrix_1[:, 0, 0] = 1.0
            # shearing_matrix_1[:, 3, 3] = 1.0
            #
            # shear_2 = shear[:, 1].view(-1)
            # shearing_matrix_2 = torch.zeros([b, 4, 4], dtype=torch.float).to(self.device)
            # shearing_matrix_2[:, 0, 0] = torch.cos(shear_2)
            # shearing_matrix_2[:, 0, 2] = torch.sin(shear_2)
            # shearing_matrix_2[:, 2, 0] = -torch.sin(shear_2)
            # shearing_matrix_2[:, 2, 2] = torch.cos(shear_2)
            # shearing_matrix_2[:, 1, 1] = 1.0
            # shearing_matrix_2[:, 3, 3] = 1.0
            #
            # shear_3 = shear[:, 2].view(-1)
            # shearing_matrix_3 = torch.zeros([b, 4, 4], dtype=torch.float).to(self.device)
            # shearing_matrix_3[:, 0, 0] = torch.cos(shear_3)
            # shearing_matrix_3[:, 0, 1] = -torch.sin(shear_3)
            # shearing_matrix_3[:, 1, 0] = torch.sin(shear_3)
            # shearing_matrix_3[:, 1, 1] = torch.cos(shear_3)
            # shearing_matrix_3[:, 2, 2] = 1.0
            # shearing_matrix_3[:, 3, 3] = 1.0
            #
            # shearing_matrix = torch.bmm(shearing_matrix_1, shearing_matrix_2)
            # shearing_matrix = torch.bmm(shearing_matrix, shearing_matrix_3)

            # Affine transform
            # matrix = torch.bmm(shearing_matrix, scaling_matrix)
            # matrix = torch.bmm(matrix, torch.transpose(shearing_matrix, 1, 2))
            # matrix = torch.bmm(matrix, rotation_matrix)
            # matrix = torch.bmm(matrix, translation_matrix)

            # Linear transform - no shearing
            matrix = torch.bmm(scaling_matrix, rotation_matrix)
            matrix = torch.bmm(matrix, translation_matrix)

            self.theta = matrix
            self.theta_inv = torch.inverse(matrix)

        def warp_image(self, img, interpolation='bilinear', padding='border'):
            wrp = self.transform(img, self.T, interpolation=interpolation, padding=padding)
            return wrp

        def warp_inv_image(self, img, interpolation='bilinear', padding='border'):
            wrp = self.transform(img, self.T_inv, interpolation=interpolation, padding=padding)
            return wrp


class ITN3D(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(self, input_size, input_nc=3, ndf=16, n_layers=3):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(ITN3D, self).__init__()
        self.device ='cpu'
        self.dtype = torch.cuda.float if (self.device == 'cuda') else torch.float

        norm_layer = nn.BatchNorm3d
        num_features = torch.prod(torch.floor_divide(torch.tensor(input_size), 32))
        self.register_buffer('grid', self.get_normalized_grid([input_size[0]//16, input_size[1]//16, input_size[2]//16]))

        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm3d
        else:
            use_bias = norm_layer != nn.BatchNorm3d

        kw = 3
        padw = 1
        sequence = [nn.Conv3d(input_nc, input_nc, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True),
                    nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=1, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                #norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
                nn.Conv3d(ndf * nf_mult, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                # norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            #norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        #sequence += [
         #   nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)


        self.fc = nn.Linear(ndf * nf_mult * num_features, ndf * nf_mult)
        self.translation = nn.Linear(ndf * nf_mult, 3)
        self.rotation = nn.Linear(ndf * nf_mult, 3)
        self.scaling = nn.Linear(ndf * nf_mult, 3)
        # self.shearing = nn.Linear(32, 3).to(self.device)

        self.translation.weight.data.zero_()
        self.translation.bias.data.copy_(torch.tensor([0, 0, 0], dtype=self.dtype))
        self.rotation.weight.data.zero_()
        self.rotation.bias.data.copy_(torch.tensor([0, 0, 0], dtype=self.dtype))
        self.scaling.weight.data.zero_()
        self.scaling.bias.data.copy_(torch.tensor([0, 0, 0], dtype=self.dtype))

    def forward(self, input):

        x5 = self.main(input)
        #x5 = self.down(x)
        xa = F.avg_pool3d(x5, 2)
        print(xa.shape)
        xa = xa.view(xa.size(0), -1)
        xa = F.relu(self.fc(xa))
        self.affine_matrix(xa)

        b, c, d, h, w = x5.size()
        id_grid = self.grid.unsqueeze(0).repeat(b, 1, 1, 1, 1)
        id_grid = id_grid.view(b, 3, -1)

        ones = torch.ones([b, 1, id_grid.size(2)]).to(self.device)

        self.T = torch.bmm(self.theta[:, 0:3, :], torch.cat((id_grid, ones), dim=1))
        self.T = self.T.view(b, 3, d, h, w)
        self.T = self.move_grid_dims(self.T)

        """Standard forward."""
        return  self.warp_image

    def get_normalized_grid(self, size):
        ndim = len(size)
        grid = torch.stack(torch.meshgrid([torch.arange(s) for s in size]), 0).float()
        for i in range(ndim):
            grid[i] = 2. * grid[i] / (size[i] - 1) - 1.

        return grid

    def regularizer(self):
        return torch.tensor([0], dtype=self.dtype).to(self.device)

    def get_T(self):
        return self.T

    def get_T_inv(self):
        return self.T_inv

    def moveaxis(self, x, src, dst):
        ndims = x.dim()
        dims = list(range(ndims))
        dims.pop(src)
        if dst < 0:
            dst = ndims + dst
        dims.insert(dst, src)

        return x.permute(dims)

    def move_grid_dims(self, grid):
        size = grid.shape[2:]
        ndim = len(size)
        assert ndim == grid.shape[1]

        grid = self.moveaxis(grid, 1, -1)  # [batch, z?, y, x, dims]
        grid = grid[..., list(range(ndim))[::-1]]  # reverse dims

        return grid

    def transform(self, src, grid, interpolation='bilinear', padding='border'):
        return F.grid_sample(src, grid, mode=interpolation, align_corners=True, padding_mode=padding)

    def affine_matrix(self, x):
        b = x.size(0)

        trans = torch.tanh(self.translation(x)) * 1.0
        translation_matrix = torch.zeros([b, 4, 4], dtype=torch.float).to(self.device)
        translation_matrix[:, 0, 0] = 1.0
        translation_matrix[:, 1, 1] = 1.0
        translation_matrix[:, 2, 2] = 1.0
        translation_matrix[:, 0, 3] = trans[:, 0].view(-1)
        translation_matrix[:, 1, 3] = trans[:, 1].view(-1)
        translation_matrix[:, 2, 3] = trans[:, 2].view(-1)
        translation_matrix[:, 3, 3] = 1.0

        rot = torch.tanh(self.rotation(x)) * (math.pi / 4.0)
        # rot Z
        angle_1 = rot[:, 0].view(-1)
        rotation_matrix_1 = torch.zeros([b, 4, 4], dtype=torch.float).to(self.device)
        rotation_matrix_1[:, 0, 0] = torch.cos(angle_1)
        rotation_matrix_1[:, 0, 1] = -torch.sin(angle_1)
        rotation_matrix_1[:, 1, 0] = torch.sin(angle_1)
        rotation_matrix_1[:, 1, 1] = torch.cos(angle_1)
        rotation_matrix_1[:, 2, 2] = 1.0
        rotation_matrix_1[:, 3, 3] = 1.0
        # rot X
        angle_2 = rot[:, 1].view(-1)
        rotation_matrix_2 = torch.zeros([b, 4, 4], dtype=torch.float).to(self.device)
        rotation_matrix_2[:, 1, 1] = torch.cos(angle_2)
        rotation_matrix_2[:, 1, 2] = -torch.sin(angle_2)
        rotation_matrix_2[:, 2, 1] = torch.sin(angle_2)
        rotation_matrix_2[:, 2, 2] = torch.cos(angle_2)
        rotation_matrix_2[:, 0, 0] = 1.0
        rotation_matrix_2[:, 3, 3] = 1.0
        # rot Z
        angle_3 = rot[:, 2].view(-1)
        rotation_matrix_3 = torch.zeros([b, 4, 4], dtype=torch.float).to(self.device)
        rotation_matrix_3[:, 0, 0] = torch.cos(angle_3)
        rotation_matrix_3[:, 0, 1] = -torch.sin(angle_3)
        rotation_matrix_3[:, 1, 0] = torch.sin(angle_3)
        rotation_matrix_3[:, 1, 1] = torch.cos(angle_3)
        rotation_matrix_3[:, 2, 2] = 1.0
        rotation_matrix_3[:, 3, 3] = 1.0

        rotation_matrix = torch.bmm(rotation_matrix_1, rotation_matrix_2)
        rotation_matrix = torch.bmm(rotation_matrix, rotation_matrix_3)

        scale = torch.tanh(self.scaling(x)) * 1.0
        scaling_matrix = torch.zeros([b, 4, 4], dtype=torch.float).to(self.device)
        scaling_matrix[:, 0, 0] = torch.exp(scale[:, 0].view(-1))
        scaling_matrix[:, 1, 1] = torch.exp(scale[:, 1].view(-1))
        scaling_matrix[:, 2, 2] = torch.exp(scale[:, 2].view(-1))
        scaling_matrix[:, 3, 3] = 1.0

        # shear = torch.tanh(self.shearing(x)) * (math.pi / 4.0)
        #
        # shear_1 = shear[:, 0].view(-1)
        # shearing_matrix_1 = torch.zeros([b, 4, 4], dtype=torch.float).to(self.device)
        # shearing_matrix_1[:, 1, 1] = torch.cos(shear_1)
        # shearing_matrix_1[:, 1, 2] = -torch.sin(shear_1)
        # shearing_matrix_1[:, 2, 1] = torch.sin(shear_1)
        # shearing_matrix_1[:, 2, 2] = torch.cos(shear_1)
        # shearing_matrix_1[:, 0, 0] = 1.0
        # shearing_matrix_1[:, 3, 3] = 1.0
        #
        # shear_2 = shear[:, 1].view(-1)
        # shearing_matrix_2 = torch.zeros([b, 4, 4], dtype=torch.float).to(self.device)
        # shearing_matrix_2[:, 0, 0] = torch.cos(shear_2)
        # shearing_matrix_2[:, 0, 2] = torch.sin(shear_2)
        # shearing_matrix_2[:, 2, 0] = -torch.sin(shear_2)
        # shearing_matrix_2[:, 2, 2] = torch.cos(shear_2)
        # shearing_matrix_2[:, 1, 1] = 1.0
        # shearing_matrix_2[:, 3, 3] = 1.0
        #
        # shear_3 = shear[:, 2].view(-1)
        # shearing_matrix_3 = torch.zeros([b, 4, 4], dtype=torch.float).to(self.device)
        # shearing_matrix_3[:, 0, 0] = torch.cos(shear_3)
        # shearing_matrix_3[:, 0, 1] = -torch.sin(shear_3)
        # shearing_matrix_3[:, 1, 0] = torch.sin(shear_3)
        # shearing_matrix_3[:, 1, 1] = torch.cos(shear_3)
        # shearing_matrix_3[:, 2, 2] = 1.0
        # shearing_matrix_3[:, 3, 3] = 1.0
        #
        # shearing_matrix = torch.bmm(shearing_matrix_1, shearing_matrix_2)
        # shearing_matrix = torch.bmm(shearing_matrix, shearing_matrix_3)

        # Affine transform
        # matrix = torch.bmm(shearing_matrix, scaling_matrix)
        # matrix = torch.bmm(matrix, torch.transpose(shearing_matrix, 1, 2))
        # matrix = torch.bmm(matrix, rotation_matrix)
        # matrix = torch.bmm(matrix, translation_matrix)

        # Linear transform - no shearing
        matrix = torch.bmm(scaling_matrix, rotation_matrix)
        matrix = torch.bmm(matrix, translation_matrix)

        self.theta = matrix
        self.theta_inv = torch.inverse(matrix)

    def warp_image(self, img, interpolation='bilinear', padding='border'):
        wrp = self.transform(img, self.T, interpolation=interpolation, padding=padding)
        return wrp

    def warp_inv_image(self, img, interpolation='bilinear', padding='border'):
        wrp = self.transform(img, self.T_inv, interpolation=interpolation, padding=padding)
        return wrp


class VQenc(nn.Module):

    def __init__(self, num_classes, inputchannels = 1, channels = 16, dropout = 0.0, n_embed = 1024,
                 embed_dim = 256):
        super(VQenc, self).__init__()
        use_bias = True
        self.conv11 = nn.Conv3d(inputchannels, channels, kernel_size=3, padding=1, stride=1, bias=use_bias)
        self.conv12 = ResnetBlock(in_channels=channels, out_channels=channels,  dropout=dropout)
        self.down1 = nn.Conv3d(channels, channels*2, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv21 = ResnetBlock(in_channels=channels*2, out_channels=channels*2,  dropout=dropout)
        self.down2 = nn.Conv3d(channels*2, channels*4, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv31 = ResnetBlock(in_channels=channels*4, out_channels=channels*4,  dropout=dropout)
        self.down3 = nn.Conv3d(channels*4, channels*8, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv41 = ResnetBlock(in_channels=channels*8, out_channels=channels*8, dropout=dropout)
        self.down4 = nn.Conv3d(channels*8, channels*16, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv51 = ResnetBlock(in_channels=channels*16, out_channels=channels*16, dropout=dropout)
        self.quant_conv = torch.nn.Conv3d(channels*16, embed_dim, 1)
        self.quantize = VectorQuantizer2(n_embed, channels*16, beta=0.25)

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
        quant, emb_loss, info = self.quantize(x5)


        return emb_loss, quant, x1, x2, x3, x4

class VQdec(nn.Module):

    def __init__(self, num_classes, inputchannels = 1, channels = 16, dropout = 0.0, n_embed = 1024,
                 embed_dim = 256):
        super(VQdec, self).__init__()

        use_bias = True

        self.post_quant_conv = torch.nn.Conv3d(embed_dim, channels*16, 1)
        self.conv52 = ResnetBlock(in_channels=channels*16, out_channels=channels*16,dropout=dropout)
        self.up4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv42 = ResnetBlock(in_channels=channels*24, out_channels=channels*8, dropout=dropout)
        self.up3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv32 = ResnetBlock(in_channels=channels*12, out_channels=channels*4, dropout=dropout)
        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv22 = ResnetBlock(in_channels=channels*6, out_channels=channels*2, dropout=dropout)
        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv13 = ResnetBlock(in_channels=channels*3, out_channels=channels, dropout=dropout)
        self.conv14 = nn.Conv3d(channels, num_classes, kernel_size=1, padding=0, bias=use_bias)

    def forward(self, quant, x1, x2, x3, x4):

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

        return x
