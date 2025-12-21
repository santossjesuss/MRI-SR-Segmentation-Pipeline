import torch.nn as nn
import math
import subpixel_block

class SubPixelUpsampler(nn.Module):
    def __init__(self, channels, upscale_factor, kernel_size=3, padding=1):
        super(SubPixelUpsampler, self).__init__()
        self.upscale_factor = upscale_factor
        self.upsample_stages = self._generate_upsample_stages(channels, kernel_size)

    def _generate_upsample_stages(self, channels, kernel_size):
        num_stages = int(math.log2(self.upscale_factor))
        stages = []
        for _ in range(num_stages):
            stages.append(subpixel_block.SubPixelBlock(channels, kernel_size=kernel_size))
        
        return nn.Sequential(*stages)

    def forward(self, x):
        return self.upsample_stages(x)