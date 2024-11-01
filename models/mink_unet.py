# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.

import os, sys
sys.path.append(os.path.join(os.getcwd(), 'lib')) # HACK add the lib folder
# from collections import OrderedDict
import MinkowskiEngine as ME
from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck
from models.resnet_base import ResNetBase
from torch import nn
from lib.config import CONF



class MinkUNetBase(ResNetBase):
    BLOCK = None
    PLANES = None
    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    INIT_DIM = 32
    OUT_TENSOR_STRIDE = 1

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling
    # initialize_coords
    def __init__(self, in_channels, out_channels, D=3):
        ResNetBase.__init__(self, in_channels, out_channels, D)

    def network_initialization(self, in_channels, out_channels, D):
        # Output of the first conv concated to conv6
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=5, dimension=D)

        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        self.conv1p1s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0],
                                       self.LAYERS[0])

        self.conv2p2s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn2 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1],
                                       self.LAYERS[1])

        self.conv3p4s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)

        self.bn3 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2],
                                       self.LAYERS[2])

        self.conv4p8s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3],
                                       self.LAYERS[3])

        self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[4], kernel_size=2, stride=2, dimension=D)
        self.bntr4 = ME.MinkowskiBatchNorm(self.PLANES[4])

        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[4],
                                       self.LAYERS[4])
        self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[5], kernel_size=2, stride=2, dimension=D)
        self.bntr5 = ME.MinkowskiBatchNorm(self.PLANES[5])

        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[5],
                                       self.LAYERS[5])
        self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[6], kernel_size=2, stride=2, dimension=D)
        self.bntr6 = ME.MinkowskiBatchNorm(self.PLANES[6])

        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
        self.block7 = self._make_layer(self.BLOCK, self.PLANES[6],
                                       self.LAYERS[6])
        self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[7], kernel_size=2, stride=2, dimension=D)
        self.bntr7 = ME.MinkowskiBatchNorm(self.PLANES[7])

        self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.block8 = self._make_layer(self.BLOCK, self.PLANES[7],
                                       self.LAYERS[7])

        self.final = ME.MinkowskiConvolution(
            self.PLANES[7],
            out_channels,
            kernel_size=1,
            # has_bias=True,
            dimension=D)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, data_dict):       # Input [B, in_channels, H, W, D]
        x = data_dict['openscene_in']    

        out = self.conv0p1s1(x)         # [B, INIT_DIM(32), H, W, D]
        out = self.bn0(out)
        out_p1 = self.relu(out)         

        out = self.conv1p1s2(out_p1)    # [B, INIT_DIM(32), H/2, W/2, D/2]
        out = self.bn1(out)
        out = self.relu(out)            
        out_b1p2 = self.block1(out)     # [B, 32, H/2, W/2, D/2]

        out = self.conv2p2s2(out_b1p2)  # [B, 32, H/4, W/4, D/4]
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)     # [B, 64, H/4, W/4, D/4]

        out = self.conv3p4s2(out_b2p4)  # [B, 64, H/8, W/8, D/8]
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)     # [B, 128, H/8, W/8, D/8]

        # tensor_stride=16
        out = self.conv4p8s2(out_b3p8)  # [B, 128, H/16, W/16, D/16]
        out = self.bn4(out)
        out = self.relu(out)
        out = self.block4(out)          # [B, 256, H/16, W/16, D/16]: Bottleneck
        data_dict['feat_bottleneck'] = out

        # tensor_stride=8
        out = self.convtr4p16s2(out)    # [B, 128, H/8, W/8, D/8]
        out = self.bntr4(out)
        out = self.relu(out)

        out = ME.cat(out, out_b3p8)
        out = self.block5(out)          # [B, 128, H/8, W/8, D/8]
        data_dict['feat_layer5'] = out

        # tensor_stride=4
        out = self.convtr5p8s2(out)     # [B, 128, H/4, W/4, D/4]
        out = self.bntr5(out)
        out = self.relu(out)

        out = ME.cat(out, out_b2p4)
        out = self.block6(out)          # [B, 96, H/4, W/4, D/4]
        data_dict['feat_layer6'] = out

        if CONF.OPENSCENE.final_result:
            # tensor_stride=2
            out = self.convtr6p4s2(out)     # [B, 96, H/2, W/2, D/2]
            out = self.bntr6(out)
            out = self.relu(out)

            out = ME.cat(out, out_b1p2)
            out = self.block7(out)          # [B, 96, H/2, W/2, D/2]

            # tensor_stride=1
            out = self.convtr7p2s2(out)     # [B, 96, H, W, D]
            out = self.bntr7(out)
            out = self.relu(out)

            out = ME.cat(out, out_p1)
            out = self.block8(out)          # [B, 96, H, W, D]

            data_dict['openscene_out'] = self.final(out).F  # [B, out_channels, H, W, D]

        return data_dict        

class MinkUNet14(MinkUNetBase):
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)


class MinkUNet18(MinkUNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)


class MinkUNet34(MinkUNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


class MinkUNet50(MinkUNetBase):
    BLOCK = Bottleneck
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


class MinkUNet101(MinkUNetBase):
    BLOCK = Bottleneck
    LAYERS = (2, 3, 4, 23, 2, 2, 2, 2)


class MinkUNet14A(MinkUNet14):
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class MinkUNet14B(MinkUNet14):
    PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class MinkUNet14C(MinkUNet14):
    PLANES = (32, 64, 128, 256, 192, 192, 128, 128)


class MinkUNet14D(MinkUNet14):
    PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class MinkUNet18A(MinkUNet18):
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class MinkUNet18B(MinkUNet18):
    PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class MinkUNet18D(MinkUNet18):
    PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class MinkUNet34A(MinkUNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 64, 64)


class MinkUNet34B(MinkUNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 64, 32)


class MinkUNet34C(MinkUNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)


def mink_unet(in_channels=3, out_channels=20, D=3, arch='MinkUNet18A'):
    if arch == 'MinkUNet18A':
        return MinkUNet18A(in_channels, out_channels, D)
    elif arch == 'MinkUNet18B':
        return MinkUNet18B(in_channels, out_channels, D)
    elif arch == 'MinkUNet18D':
        return MinkUNet18D(in_channels, out_channels, D)
    elif arch == 'MinkUNet34A':
        return MinkUNet34A(in_channels, out_channels, D)
    elif arch == 'MinkUNet34B':
        return MinkUNet34B(in_channels, out_channels, D)
    elif arch == 'MinkUNet34C':
        return MinkUNet34C(in_channels, out_channels, D)
    elif arch == 'MinkUNet14A':
        return MinkUNet14A(in_channels, out_channels, D)
    elif arch == 'MinkUNet14B':
        return MinkUNet14B(in_channels, out_channels, D)
    elif arch == 'MinkUNet14C':
        return MinkUNet14C(in_channels, out_channels, D)
    elif arch == 'MinkUNet14D':
        return MinkUNet14D(in_channels, out_channels, D)
    else:
        raise Exception('architecture not supported yet'.format(arch))


# def state_dict_remove_moudle(state_dict):
#     new_state_dict = OrderedDict()
#     for k, v in state_dict.items():
#         name = k.replace('module.', '')
#         new_state_dict[name] = v
#     return new_state_dict


def constructor3d(**kwargs):
    model = mink_unet(**kwargs)
    return model


class DisNet(nn.Module):
    '''3D Sparse UNet for Distillation.'''
    def __init__(self, feature_2d_extractor='openseg'):
        super(DisNet, self).__init__()
        if 'lseg' in feature_2d_extractor:
            last_dim = 512
        elif 'openseg' in feature_2d_extractor:
            last_dim = 768
        else:
            raise NotImplementedError

        # MinkowskiNet for 3D point clouds
        net3d = constructor3d(in_channels=3, out_channels=last_dim, D=3, arch=CONF.OPENSCENE.arch_3d)
        self.net3d = net3d

    def forward(self, sparse_3d):
        '''Forward method.'''
        return self.net3d(sparse_3d)
