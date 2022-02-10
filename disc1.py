import functools
import torch.nn as nn
import torch
import torch.nn.functional as F
import math


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class NLayerDiscriminator2D(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator2D, self).__init__()

        norm_layer = nn.BatchNorm2d


        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)

class NLayerDiscriminator3D(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator3D, self).__init__()

        norm_layer = nn.BatchNorm3d


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

    def forward(self, input):
        """Standard forward."""
        return self.main(input)

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

            self.device = 'cuda'
            # self.dtype = torch.cuda.float if (self.device == 'cuda') else torch.float
            self.dtype = torch.float

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
        self.device ='cuda'
        #self.dtype = torch.cuda.float if (self.device == 'cuda') else torch.float
        self.dtype = torch.float

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
