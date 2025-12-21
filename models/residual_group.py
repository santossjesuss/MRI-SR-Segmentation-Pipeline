import torch.nn as nn
from nodels.residual_channel_attention_block import RCAB

class ResidualGroup(nn.Module):
    def __init__(self, channels, num_blocks):
        super(ResidualGroup, self).__init__()
        blocks = [RCAB(channels) for _ in range(num_blocks)]
        self.stacked_rcabs = nn.Sequential(*blocks)
        self.final_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        rcabs_features = self.stacked_rcabs(x)
        rg_features = self.final_conv(rcabs_features)
        return x + rg_features