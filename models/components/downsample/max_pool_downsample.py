from .downsample import Downsample
import torch.nn as nn

class MaxPoolDownsample(Downsample):
    def __init__(self, kernel_size, stride):
        super(MaxPoolDownsample, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

        self.pool = nn.MaxPool2d(kernel_size=self.kernel_size, stride=self.stride)

    def forward(self, x):
        return self.pool(x)