import torch
import torch.nn as nn
from .double_conv_block import DoubleConvolution as DoubleConv
from ..components.upsample.upsample import Upsample

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, kernel_size, padding, stride, upsample: Upsample = None):
        super(DecoderBlock, self).__init__()
        concatenated_channels = skip_channels + out_channels
        self.upsample = self._choose_upsampler(in_channels, out_channels, kernel_size, stride, upsample)
        self.double_conv = DoubleConv(concatenated_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def _choose_upsampler(self, in_channels, out_channels, kernel_size, stride, upsample: Upsample):
        if upsample is not None:
            return upsample
        else:
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x, skip_connection):
        upsampled = self.upsample(x)
        concatenated = torch.cat((upsampled, skip_connection), dim=1)
        decoded = self.double_conv(concatenated)
        return decoded