
# torchvisison >= 0.9.0
import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision.ops import deform_conv2d
import math
import logging
logger = logging.getLogger('base')

class ModulatedDeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 bias=True,
                 deformable_groups=1,
                 extra_offset_mask=True, 
                 offset_in_channel=32
                 ):
        super(ModulatedDeformableConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride 
        self.padding = padding 
        self.dilation = dilation
        self.groups = groups
        self.extra_offset_mask = extra_offset_mask

        self.conv_offset_mask = nn.Conv2d(offset_in_channel,
                                     deformable_groups * 3 * kernel_size * kernel_size,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     bias=True)

        self.init_offset()

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None
        self.init_weights()

    def init_weights(self):
        n = self.in_channels * self.kernel_size * self.kernel_size
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()
    
    def init_offset(self):
        torch.nn.init.constant_(self.conv_offset_mask.weight, 0.)
        torch.nn.init.constant_(self.conv_offset_mask.bias, 0.)


    def forward(self, x):

        if self.extra_offset_mask:
            # x = [input, features]
            offset_mask = self.conv_offset_mask(x[1])
            x = x[0]
        else:
            offset_mask = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        offset_mean = torch.mean(torch.abs(offset))
        if offset_mean > max(x.shape[2:]):
            logger.warning('Offset mean is {}, larger than max(h, w).'.format(offset_mean))

        out = deform_conv2d(input=x,
                            offset=offset,
                            weight=self.weight,
                            bias=self.bias,
                            stride=self.stride,
                            padding=self.padding,
                            dilation=self.dilation,
                            mask=mask
                            )


        return out
