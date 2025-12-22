import torch.nn as nn
from attention.attention_factory import AttentionFactory
    
# Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, channels, kernel_size, padding, bias=True, attention_module="eca"):
        super(RCAB, self).__init__()
        self.convolutions = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, bias=bias)
        )
        self.channel_attention = AttentionFactory.get_attention_module(attention_module, channels=channels)

    def forward(self, x):
        features = self.convolutions(x)
        attention_weights = self.channel_attention(features)
        weighted_features = features * attention_weights
        return x + weighted_features