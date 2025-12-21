import torch.nn as nn
import math

# Efficient Channel Attention (ECA) Block
class ECA_Block(nn.Module):
    def __init__(self, channels):
        super(ECA_Block, self).__init__()
        kernel_size = self.calculate_adaptive_kernel_size(channels)
        padding = (kernel_size - 1) // 2
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.cross_channel_conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.gap(x)
        y = y.squeeze(-1).transpose(1, 2)
        y = self.cross_channel_conv(y)
        y = self.sigmoid(y)
        y = y.transpose(1, 2).unsqueeze(-1)
        scaled_y = y.expand_as(x)
        return x * scaled_y
    
    def calculate_adaptive_kernel_size(channels, gamma=2, b=1):
        kernel_size = int(abs(math.log2(channels) / gamma + b / gamma))
        return kernel_size if kernel_size % 2 == 1 else kernel_size + 1