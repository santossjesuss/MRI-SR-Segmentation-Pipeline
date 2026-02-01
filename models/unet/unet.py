import torch
import torch.nn as nn
from .double_conv_block import DoubleConvolution as DoubleConv
from .encoder_block import EncoderBlock
from ..components.downsample.downsample import Downsample
from ..components.downsample.max_pool_downsample import MaxPoolDownsample
from .decoder_block import DecoderBlock

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=64, kernel_size=3):
        super(UNet, self).__init__()
        self._validate_args(in_channels, out_channels, base_channels, kernel_size)
        # self.conv_kernel_size = kernel_size
        # self.upsample_kernel_size = 2
        self.kernel_size = kernel_size
        self.pool_kernel_size = 2
        self.padding = self.kernel_size // 2
        self.downsample_stride = 2
        self.upsample_stride = 2

        self.downsample: Downsample = MaxPoolDownsample(kernel_size=self.pool_kernel_size, stride=self.downsample_stride)
        
        self.encoder1 = EncoderBlock(in_channels, base_channels, kernel_size=self.kernel_size, padding=self.padding, downsample=self.downsample)
        self.encoder2 = EncoderBlock(base_channels, base_channels * 2, kernel_size=self.kernel_size, padding=self.padding, downsample=self.downsample)
        self.encoder3 = EncoderBlock(base_channels * 2, base_channels * 4, kernel_size=self.kernel_size, padding=self.padding, downsample=self.downsample)
        self.encoder4 = EncoderBlock(base_channels * 4, base_channels * 8, kernel_size=self.kernel_size, padding=self.padding, downsample=self.downsample)
        
        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16, kernel_size=self.kernel_size, padding=self.padding)
        
        self.decoder4 = DecoderBlock(base_channels * 16, base_channels * 8, base_channels * 8, kernel_size=self.kernel_size, padding=self.padding, stride=self.upsample_stride)
        self.decoder3 = DecoderBlock(base_channels * 8, base_channels * 4, base_channels * 4, kernel_size=self.kernel_size, padding=self.padding, stride=self.upsample_stride)
        self.decoder2 = DecoderBlock(base_channels * 4, base_channels * 2, base_channels * 2, kernel_size=self.kernel_size, padding=self.padding, stride=self.upsample_stride)
        self.decoder1 = DecoderBlock(base_channels * 2, base_channels, base_channels, kernel_size=self.kernel_size, padding=self.padding, stride=self.upsample_stride)
        
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1, padding=0)

    def _validate_args(self, in_channels, out_channels, base_channels, kernel_size):
        if in_channels < 1:
            raise ValueError(f"Input channels must be a positive integer. Current value: {in_channels}") 
        
        if out_channels < 1:
            raise ValueError(f"Output channels must be a positive integer. Current value: {out_channels}")

        if base_channels < 1:
            raise ValueError(f"Base channels must be a positive integer. Current value: {base_channels}")

        if kernel_size < 1:
            raise ValueError(f"Kernel size must be a positive integer. Current value: {kernel_size}")

    def forward(self, x):
        skip1, down1 = self.encoder1(x)
        skip2, down2 = self.encoder2(down1)
        skip3, down3 = self.encoder3(down2)
        skip4, down4 = self.encoder4(down3)

        bottleneck_enc = self.bottleneck(down4)

        dec4 = self.decoder4(bottleneck_enc, skip_connection=skip4)
        dec3 = self.decoder3(dec4, skip_connection=skip3)
        dec2 = self.decoder2(dec3, skip_connection=skip2)
        dec1 = self.decoder1(dec2, skip_connection=skip1)

        return self.final_conv(dec1)