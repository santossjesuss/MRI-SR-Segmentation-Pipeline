from metrics.base_metrics import BaseMetrics
from torchmetrics.segmentation import DiceScore, MeanIoU
from torchmetrics.classification import Precision, Recall, MulticlassF1Score, MulticlassJaccardIndex, MulticlassPrecision, MulticlassRecall

class SegmentationMetrics(BaseMetrics):
    def __init__(self, num_classes, include_background, ignore_index):
        super().__init__()
        self._init_mean_metrics(num_classes, include_background, ignore_index)
        # self._init_listed_metrics(num_classes, include_background, ignore_index)
        # self._init_mean_multiclass_metrics(num_classes, ignore_index)

    def update(self, predicted_masks, true_masks):
        for metric in self.metrics.values():
            metric.update(predicted_masks, true_masks)

    def _init_mean_metrics(self, num_classes, include_background, ignore_index):
        self.metrics['dice'] = DiceScore(
            num_classes=num_classes, 
            average='macro', 
            include_background=include_background    
        )
        self.metrics['iou'] = MeanIoU(
            num_classes=num_classes, 
            include_background=include_background
        )
        self.metrics['precision'] = Precision(
            task='multiclass', 
            num_classes=num_classes, 
            average='macro', 
            ignore_index=ignore_index
        )
        self.metrics['recall'] = Recall(
            task='multiclass', 
            num_classes=num_classes, 
            average='macro', 
            ignore_index=ignore_index
        )

    def _init_listed_metrics(self, num_classes, include_background, ignore_index):
        self.metrics['dice'] = DiceScore(
            num_classes=num_classes, 
            average=None, 
            include_background=include_background
        )
        self.metrics['iou'] = MeanIoU(
            num_classes=num_classes, 
            include_background=include_background
        )
        self.metrics['precision'] = Precision(
            task='multiclass', 
            num_classes=num_classes, 
            average=None, 
            ignore_index=ignore_index
        )
        self.metrics['recall'] = Recall(
            task='multiclass', 
            num_classes=num_classes, 
            average=None, 
            ignore_index=ignore_index
        )

    def _init_mean_multiclass_metrics(self, num_classes, ignore_index):
        self.metrics['dice'] = MulticlassF1Score(
            num_classes=num_classes, 
            average='macro', 
            ignore_index=ignore_index
        )
        self.metrics['iou'] = MulticlassJaccardIndex(
            num_classes=num_classes, 
            ignore_index=ignore_index
        )
        self.metrics['precision'] = MulticlassPrecision(
            num_classes=num_classes, 
            average='macro', 
            ignore_index=ignore_index
        )
        self.metrics['recall'] = MulticlassRecall(
            num_classes=num_classes, 
            average='macro', 
            ignore_index=ignore_index
        )