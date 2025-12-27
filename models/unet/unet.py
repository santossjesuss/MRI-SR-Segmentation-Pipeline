import torch
import torch.nn as nn
from .double_conv_block import DoubleConvolution as DoubleConv
from ..components.downsample.downsample import Downsample
from ..components.downsample.max_pool_downsample import MaxPoolDownsample

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=64, kernel_size=3):
        super(UNet, self).__init__()
        self._validate_args(in_channels, out_channels, base_channels, kernel_size)
        self.conv_kernel_size = kernel_size
        self.upsample_kernel_size = 2
        self.pool_kernel_size = 2
        self.padding = self.conv_kernel_size // 2
        self.downsample_stride = 2
        self.upsample_stride = 2
        self.downsample: Downsample = MaxPoolDownsample(kernel_size=self.pool_kernel_size, stride=self.downsample_stride)

        self.pool = self.downsample()
        self.encoder1 = DoubleConv(in_channels, base_channels, kernel_size=self.conv_kernel_size, padding=self.padding)
        self.encoder2 = DoubleConv(base_channels, base_channels * 2, kernel_size=self.conv_kernel_size, padding=self.padding)
        self.encoder3 = DoubleConv(base_channels * 2, base_channels * 4, kernel_size=self.conv_kernel_size, padding=self.padding)
        self.encoder4 = DoubleConv(base_channels * 4, base_channels * 8, kernel_size=self.conv_kernel_size, padding=self.padding)
        
        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16, kernel_size=self.conv_kernel_size, padding=self.padding)
        
        self.upsample4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, self.upsample_kernel_size, stride=self.upsample_stride)
        self.decoder4 = DoubleConv(base_channels * 16, base_channels * 8, kernel_size=self.conv_kernel_size, padding=self.padding)
        self.upsample3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, self.upsample_kernel_size, stride=self.upsample_stride)
        self.decoder3 = DoubleConv(base_channels * 8, base_channels * 4, kernel_size=self.conv_kernel_size, padding=self.padding)
        self.upsample2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, self.upsample_kernel_size, stride=self.upsample_stride)
        self.decoder2 = DoubleConv(base_channels * 4, base_channels * 2, kernel_size=self.conv_kernel_size, padding=self.padding)
        self.upsample1 = nn.ConvTranspose2d(base_channels * 2, base_channels, self.upsample_kernel_size, stride=self.upsample_stride)
        self.decoder1 = DoubleConv(base_channels * 2, base_channels, kernel_size=self.conv_kernel_size, padding=self.padding)
        
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
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))

        bottleneck = self.bottleneck(self.pool(enc4))

        up4 = self.upsample4(bottleneck)
        dec4 = self.decoder4(torch.cat([up4, enc4], dim=1))

        up3 = self.upsample3(dec4)
        dec3 = self.decoder3(torch.cat([up3, enc3], dim=1))

        up2 = self.upsample2(dec3)
        dec2 = self.decoder2(torch.cat([up2, enc2], dim=1))

        up1 = self.upsample1(dec2)
        dec1 = self.decoder1(torch.cat([up1, enc1], dim=1))

        return self.final_conv(dec1)