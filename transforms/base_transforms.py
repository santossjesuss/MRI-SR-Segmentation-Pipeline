import torch.nn.functional as F

class BaseTransforms():
    def __init__(self, scale_factor):
        self.scale_factor = scale_factor
        self.downsample_factor = self._calc_downsample_factor(scale_factor)
        self.img_downsample_mode = 'bicubic'
        self.mask_downsample_mode = 'nearest'

    def normalize_image(self, image):
        return image.float() / 255.0

    def normalize_binary_mask(self, mask):
        return (mask > 0).long()

    def downsample_image(self, image):
        image = image.unsqueeze(0)
        lr_image = F.interpolate(image, scale_factor=self.downsample_factor, mode=self.img_downsample_mode)

        return lr_image.squeeze(0)

    def downsample_mask(self, mask):
        mask = mask.unsqueeze(0)
        lr_mask = F.interpolate(mask, scale_factor=self.downsample_factor, mode=self.mask_downsample_mode)

        return lr_mask.squeeze(0)
    
    def _calc_downsample_factor(self, scale_factor):
        return 1 / scale_factor