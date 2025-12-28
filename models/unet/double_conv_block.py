import torch.nn as nn

class DoubleConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, mid_channels=None):
        super(DoubleConvolution, self).__init__()
        self.group_norm_size = self._calc_group_normalization_size(out_channels)
        mid_channels = self._get_middle_channels(mid_channels, out_channels)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size, padding),
            nn.GroupNorm(num_groups=self.group_norm_size, num_channels=mid_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size, padding),
            nn.GroupNorm(num_groups=self.group_norm_size, num_channels=out_channels),
            nn.ReLU(inplace=True)
        )

    def _get_middle_channels(self, mid_channels, out_channels):
        return mid_channels if mid_channels is not None else out_channels
    
    def _calc_group_normalization_size(self, channels):
        # for group in [8, 4, 2, 1]:
        #     if channels % group == 0:
        #         return group
        return 4

    def forward(self, x):
        features_1 = self.conv1(x)
        features_2 = self.conv2(features_1)
        return features_2