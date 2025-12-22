import torch
import torch.nn as nn

class SubPixelBlock(nn.Module):
    def __init__(self, channels, kernel_size, padding):
        super(SubPixelBlock, self).__init__()
        self.upscale_factor = 2
        out_channels = self._calc_out_channels(channels)

        self.upsample_conv = nn.Conv2d(channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.pixel_shuffle = nn.PixelShuffle(self.upscale_factor)
        self.relu = nn.ReLU(inplace=True)

        self._icnr_init()

    def _calc_out_channels(self, channels):
        return channels * (self.upscale_factor ** 2)

    def _icnr_init(self):
        weight = self.upsample_conv.weight
        out_channels, in_channels, kH, kW = weight.shape
        r2 = self.upscale_factor ** 2
        # assert out_channels % r2 == 0

        subkernel = torch.zeros(
            out_channels // r2, in_channels, kH, kW, device = weight.device
        )
        nn.init.kaiming_normal_(subkernel)
        subkernel = subkernel.repeat(r2, 1, 1, 1)

        with torch.no_grad():
            weight.copy_(subkernel)

    def forward(self, x):
        x = self.upsample_conv(x)
        x = self.pixel_shuffle(x)
        return self.relu(x)