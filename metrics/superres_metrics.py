import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from metrics.base_metrics import BaseMetrics

class SuperResolutionMetrics(BaseMetrics):
    def __init__(self):
        super().__init__()
        self.metrics['psnr'] = PeakSignalNoiseRatio(data_range=1.0)
        self.metrics['ssim'] = StructuralSimilarityIndexMeasure(data_range=1.0)

    def update(self, sr_images, hr_images):
        sr_images = torch.clamp(sr_images, 0, 1)
        hr_images = torch.clamp(hr_images, 0, 1)

        for metric in self.metrics.values():
            metric.update(sr_images, hr_images)