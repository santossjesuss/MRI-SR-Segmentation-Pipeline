import torch.nn as nn
from math import log2
from subpixel_block import SubPixelBlock

class SubPixelUpsampler(nn.Module):
    def __init__(self, channels, upscale_factor, kernel_size, padding):
        super(SubPixelUpsampler, self).__init__()
        self.upscale_factor = upscale_factor
        self.upsample_stages = self._generate_upsample_stages(channels, kernel_size, padding)

    def _generate_upsample_stages(self, channels, kernel_size, padding):
        num_stages = int(log2(self.upscale_factor))
        stages = []
        for _ in range(num_stages):
            stages.append(SubPixelBlock(channels, kernel_size=kernel_size, padding=padding))
        
        return nn.Sequential(*stages)

    def forward(self, x):
        return self.upsample_stages(x)