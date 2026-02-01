import torch.nn as nn
from .double_conv_block import DoubleConvolution as DoubleConv
from ..components.downsample.downsample import Downsample

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, downsample: Downsample):
        super(EncoderBlock, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.pool = downsample

    def forward(self, x):
        encoded_channels = self.double_conv(x)
        downsampled_channels = self.pool(encoded_channels)
        print(f'[Encoder] Skip: {encoded_channels.shape}')
        print(f'[Encoder] Downsampled: {downsampled_channels.shape}')
        return encoded_channels, downsampled_channels