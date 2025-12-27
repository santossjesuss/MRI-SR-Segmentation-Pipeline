from .upsample import Upsample
import torch as nn

class TransposedConvUpsample(Upsample):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(TransposedConvUpsample, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=self.kernel_size, stride=self.stride)

    def forward(self, x):
        return self.conv_transpose(x)