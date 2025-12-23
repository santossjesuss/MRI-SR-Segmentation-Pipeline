import torch.nn as nn
from .residual_group import ResidualGroup
from .upsample.subpixel_upsampler import SubPixelUpsampler

# Residual Channel Attention Network (RCAN) in 2D
class RCAN(nn.Module):
    def __init__(self, num_rg=5, num_rcab=10, channels=64, kernel_size=3, upscale_factor=2):
        super(RCAN, self).__init__()
        self._validate_args(num_rg, num_rcab, channels, kernel_size, upscale_factor)
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        residual_groups = [ResidualGroup(channels, num_rcab, self.kernel_size, self.padding) for _ in range(num_rg)]

        self.shallow_feature_extractor = nn.Conv2d(3, channels, kernel_size=self.kernel_size, padding=self.padding)
        self.residual_in_residual = nn.Sequential(
            *residual_groups,
            nn.Conv2d(channels, channels, kernel_size=self.kernel_size, padding=self.padding)
        )
        self.upsample = SubPixelUpsampler(channels, upscale_factor, kernel_size=self.kernel_size, padding=self.padding)
        self.final_conv = nn.Conv2d(channels, 3, kernel_size=self.kernel_size, padding=self.padding)

    def _validate_args(self, num_rg, num_rcab, channels, kernel_size, upscale_factor):
        if num_rg < 1:
            raise ValueError(f"Residual Groups quantity must be a positive integer. Current value: {num_rg}") 
        
        if num_rcab < 1:
            raise ValueError(f"RCABs per Residual Group must be a positive integer. Current value: {num_rcab}")

        if channels < 1:
            raise ValueError(f"Channels must be a positive integer. Current value: {channels}")

        if kernel_size < 1:
            raise ValueError(f"Kernel size must be a positive integer. Current value: {kernel_size}")

        if upscale_factor < 1 or upscale_factor % 2 != 0:
            raise ValueError(f"Upscale factor must be a power of two. Current value: {upscale_factor}")

    def forward(self, x):
        shallow_features = self.shallow_feature_extractor(x)
        deep_features = self.residual_in_residual(shallow_features)
        deep_features += shallow_features
        upsampled = self.upsample(deep_features)
        return self.final_conv(upsampled)
