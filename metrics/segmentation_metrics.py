from metrics.base_metrics import BaseMetrics
from torchmetrics.segmentation import DiceScore, MeanIoU
from torchmetrics.classification import Precision, Recall

class SegmentationMetrics(BaseMetrics):
    def __init__(self, num_classes):
        super().__init__()
        self._init_mean_metrics(num_classes)
        # self._init_listed_metrics(num_classes)    # debugging

    def update(self, predicted_masks, true_masks):
        for metric in self.metrics.values():
            metric.update(predicted_masks, true_masks)

    def _init_mean_metrics(self, num_classes):
        self.metrics['dice'] = DiceScore(
            num_classes=num_classes, 
            average='macro'
        )
        self.metrics['iou'] = MeanIoU(
            num_classes=num_classes
        )
        self.metrics['precision'] = Precision(
            task='multiclass', 
            num_classes=num_classes, 
            average='macro'
        )
        self.metrics['recall'] = Recall(
            task='multiclass', 
            num_classes=num_classes, 
            average='macro'
        )

    def _init_listed_metrics(self, num_classes):
        self.metrics['dice'] = DiceScore(
            num_classes=num_classes, 
            average=None
        )
        self.metrics['iou'] = MeanIoU(
            num_classes=num_classes
        )
        self.metrics['precision'] = Precision(
            task='multiclass', 
            num_classes=num_classes, 
            average=None
        )
        self.metrics['recall'] = Recall(
            task='multiclass', 
            num_classes=num_classes, 
            average=None
        )