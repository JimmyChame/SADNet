import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
from distutils.version import LooseVersion
import torchvision
if LooseVersion(torchvision.__version__) >= LooseVersion('0.9.0'):
    from model.deform_conv import ModulatedDeformableConv2d as DCN
else:
    try:
        from dcn.deform_conv import ModulatedDeformConvPack2 as DCN
    except ImportError:
        raise ImportError('Failed to import DCNv2 module.')

#==============================================================================#
class ResBlock(nn.Module):

    def __init__(self, input_channel=32, output_channel=32):
        super().__init__()
        self.in_channel = input_channel
        self.out_channel = output_channel
        if self.in_channel != self.out_channel:
            self.conv0 = nn.Conv2d(input_channel, output_channel, 1, 1)
        self.conv1 = nn.Conv2d(output_channel, output_channel, 3, 1, 1)
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.initialize_weights()

    def forward(self, x):
        if self.in_channel != self.out_channel:
            x = self.conv0(x)
        conv1 = self.lrelu(self.conv1(x))
        conv2 = self.conv2(conv1)
        out = x + conv2
        return out
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

class RSABlock(nn.Module):

    def __init__(self, input_channel=32, output_channel=32, offset_channel=32):
        super().__init__()
        self.in_channel = input_channel
        self.out_channel = output_channel
        if self.in_channel != self.out_channel:
            self.conv0 = nn.Conv2d(input_channel, output_channel, 1, 1)
        self.dcnpack = DCN(output_channel, output_channel, 3, stride=1, padding=1, dilation=1, deformable_groups=8,
                            extra_offset_mask=True, offset_in_channel=offset_channel)
        self.conv1 = nn.Conv2d(output_channel, output_channel, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.initialize_weights()

    def forward(self, x, offset):
        if self.in_channel != self.out_channel:
            x = self.conv0(x)
        fea = self.lrelu(self.dcnpack([x, offset]))
        out = self.conv1(fea) + x
        return out
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

class OffsetBlock(nn.Module):

    def __init__(self, input_channel=32, offset_channel=32, last_offset=False):
        super().__init__()
        self.offset_conv1 = nn.Conv2d(input_channel, offset_channel, 3, 1, 1)  # concat for diff
        if last_offset:
            self.offset_conv2 = nn.Conv2d(offset_channel*2, offset_channel, 3, 1, 1)  # concat for offset
        self.offset_conv3 = nn.Conv2d(offset_channel, offset_channel, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.initialize_weights()

    def forward(self, x, last_offset=None):
        offset = self.lrelu(self.offset_conv1(x))
        if last_offset is not None:
            last_offset = F.interpolate(last_offset, scale_factor=2, mode='bilinear', align_corners=False)
            offset = self.lrelu(self.offset_conv2(torch.cat([offset, last_offset * 2], dim=1)))
        offset = self.lrelu(self.offset_conv3(offset))
        return offset
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

class ContextBlock(nn.Module):
    def __init__(self, input_channel=32, output_channel=32, square=False):
        super().__init__()
        self.conv0 = nn.Conv2d(input_channel, output_channel, 1, 1)
        if square:
            self.conv1 = nn.Conv2d(output_channel, output_channel, 3, 1, 1, 1)
            self.conv2 = nn.Conv2d(output_channel, output_channel, 3, 1, 2, 2)
            self.conv3 = nn.Conv2d(output_channel, output_channel, 3, 1, 4, 4)
            self.conv4 = nn.Conv2d(output_channel, output_channel, 3, 1, 8, 8)
        else:
            self.conv1 = nn.Conv2d(output_channel, output_channel, 3, 1, 1, 1)
            self.conv2 = nn.Conv2d(output_channel, output_channel, 3, 1, 2, 2)
            self.conv3 = nn.Conv2d(output_channel, output_channel, 3, 1, 3, 3)
            self.conv4 = nn.Conv2d(output_channel, output_channel, 3, 1, 4, 4)
        self.fusion = nn.Conv2d(4*output_channel, input_channel, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.initialize_weights()

    def forward(self, x):
        x_reduce = self.conv0(x)
        conv1 = self.lrelu(self.conv1(x_reduce))
        conv2 = self.lrelu(self.conv2(x_reduce))
        conv3 = self.lrelu(self.conv3(x_reduce))
        conv4 = self.lrelu(self.conv4(x_reduce))
        out = torch.cat([conv1, conv2, conv3, conv4], 1)
        out = self.fusion(out) + x
        return out
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


#===============================================================================#
class SADNET(nn.Module):

    def __init__(self, input_channel=3, output_channel=3, n_channel=32, offset_channel=32):
        super().__init__()

        self.res1 = ResBlock(input_channel, n_channel)
        self.down1 = nn.Conv2d(n_channel, n_channel*2, 2, 2)
        self.res2 = ResBlock(n_channel*2, n_channel*2)
        self.down2 = nn.Conv2d(n_channel*2, n_channel*4, 2, 2)
        self.res3 = ResBlock(n_channel*4, n_channel*4)
        self.down3 = nn.Conv2d(n_channel*4, n_channel*8, 2, 2)
        self.res4 = ResBlock(n_channel*8, n_channel*8)

        self.context = ContextBlock(n_channel*8, n_channel*2, square=False)
        self.offset4 = OffsetBlock(n_channel*8, offset_channel, False)
        self.dres4 = RSABlock(n_channel*8, n_channel*8, offset_channel)

        self.up3 = nn.ConvTranspose2d(n_channel*8, n_channel*4, 2, 2)
        self.dconv3_1 = nn.Conv2d(n_channel*8, n_channel*4, 1, 1)
        self.offset3 = OffsetBlock(n_channel*4, offset_channel, True)
        self.dres3 = RSABlock(n_channel*4, n_channel*4, offset_channel)

        self.up2 = nn.ConvTranspose2d(n_channel*4, n_channel*2, 2, 2)
        self.dconv2_1 = nn.Conv2d(n_channel*4, n_channel*2, 1, 1)
        self.offset2 = OffsetBlock(n_channel*2, offset_channel, True)
        self.dres2 = RSABlock(n_channel*2, n_channel*2, offset_channel)

        self.up1 = nn.ConvTranspose2d(n_channel*2, n_channel, 2, 2)
        self.dconv1_1 = nn.Conv2d(n_channel*2, n_channel, 1, 1)
        self.offset1 = OffsetBlock(n_channel, offset_channel, True)
        self.dres1 = RSABlock(n_channel, n_channel, offset_channel)

        self.out = nn.Conv2d(n_channel, output_channel, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        conv1 = self.res1(x)
        pool1 = self.lrelu(self.down1(conv1))
        conv2 = self.res2(pool1)
        pool2 = self.lrelu(self.down2(conv2))
        conv3 = self.res3(pool2)
        pool3 = self.lrelu(self.down3(conv3))
        conv4 = self.res4(pool3)
        conv4 = self.context(conv4)

        L4_offset = self.offset4(conv4, None)
        dconv4 = self.dres4(conv4, L4_offset)

        up3 = torch.cat([self.up3(dconv4), conv3], 1)
        up3 = self.dconv3_1(up3)
        L3_offset = self.offset3(up3, L4_offset)
        dconv3 = self.dres3(up3, L3_offset)

        up2 = torch.cat([self.up2(dconv3), conv2], 1)
        up2 = self.dconv2_1(up2)
        L2_offset = self.offset2(up2, L3_offset)
        dconv2 = self.dres2(up2, L2_offset)

        up1 = torch.cat([self.up1(dconv2), conv1], 1)
        up1 = self.dconv1_1(up1)
        L1_offset = self.offset1(up1, L2_offset)
        dconv1 = self.dres1(up1, L1_offset)

        out = self.out(dconv1) + x

        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                #torch.nn.init.xavier_normal_(m.weight.data)
                torch.nn.init.xavier_uniform_(m.weight.data)
                #torch.nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()
#==============================================================================#
