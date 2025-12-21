import torch.nn as nn
import attention_factory
    
# Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, channels, kernel_size=3, bias=True, attention_module="se"):
        super(RCAB, self).__init__()
        self.convolutions = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2, bias=bias)
        )
        self.channel_attention = attention_factory.get_attention_module(attention_module, channels=channels)

    def forward(self, x):
        features = self.convolutions(x)
        attention_weights = self.channel_attention(features)
        weighted_features = features * attention_weights
        return x + weighted_features