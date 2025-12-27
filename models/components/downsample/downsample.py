from abc import ABC, abstractmethod
import torch.nn as nn

class Downsample(ABC, nn.Module):
    @abstractmethod
    def forward(self, x):
        pass