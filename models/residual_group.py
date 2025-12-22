import torch.nn as nn
from residual_channel_attention_block import RCAB

class ResidualGroup(nn.Module):
    def __init__(self, channels, num_blocks, kernel_size, padding):
        super(ResidualGroup, self).__init__()
        blocks = [RCAB(channels, kernel_size, padding) for _ in range(num_blocks)]
        self.stacked_rcabs = nn.Sequential(*blocks)
        self.final_conv = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        rcabs_features = self.stacked_rcabs(x)
        rg_features = self.final_conv(rcabs_features)
        return x + rg_features