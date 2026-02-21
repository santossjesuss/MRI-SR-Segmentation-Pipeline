from torchmetrics.segmentation import DiceScore, MeanIoU
from torchmetrics.classification import Precision, Recall

class SegmentationMetrics:
    def __init__(self, num_classes, ignore_index, device):
        self.metrics = {
            'dice_mean': DiceScore(
                num_classes=num_classes, 
                average='macro', 
            ).to(device),

            'dice': DiceScore(
                num_classes=num_classes, 
                average=None, 
            ).to(device),

            'iou': MeanIoU(
                num_classes=num_classes, 
            ).to(device),

            'precision': Precision(
                task='multiclass',
                num_classes=num_classes, 
                average=None, 
                ignore_index=ignore_index
            ).to(device),

            'recall': Recall(
                task='multiclass',
                num_classes=num_classes, 
                average=None, 
                ignore_index=ignore_index
            ).to(device)
        }

    def update(self, predicted_masks, true_masks):
        for metric in self.metrics.values():
            metric.update(predicted_masks, true_masks)

    def compute(self):
        results = {}
        for name, metric in self.metrics.items():
            value = metric.compute()
            results[name] = value.item() if value.dim() == 0 else value.tolist()

        return results
    
    def reset(self):
        for metric in self.metrics.values():
            metric.reset()