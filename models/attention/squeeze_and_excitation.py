import torch.nn as nn

# Squeeze-and-Excitation (SE) Block
class SE_Block(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SE_Block, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.squeeze(x)
        y = self.excitation(y)
        return x * y


class SE_Block(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SE_Block, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        y = self.squeeze(x).view(batch_size, channels)
        y = self.excitation(y).view(batch_size, channels, 1, 1)
        scaled_y = y.expand_as(x)
        return x * scaled_y