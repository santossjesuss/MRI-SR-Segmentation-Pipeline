from datasets.mrbrains_preprocessed_dataset import MRBrainSPreprocessedDataset
from utils.segmentation_transforms import segmentation_train_transform, segmentation_validation_transform
from pipelines.segmentation_only import SegmentationOnlyPipeline
from utils.dice_ce_combined_loss import DiceCECombinedLoss

class MRBrainSUNetExperiment:
    def __init__(self):
        self.model_name = 'resnet34'
        self.in_channels = 3    # Modalities (T1, T1_IR, T2_FLAIR)
        self.classes = 9
        self.batch_size = 4
        self.shuffle_data = True
        self.learning_rate = 1e-4
        self.epochs = 50
        self.ignore_index = 0
        self.saving_name = 'mrbrains_unet.pth'
        self.criterion = DiceCECombinedLoss(dice_weight=0.5, cross_entropy_weight=0.5)

    def run(self):
        train_transform = segmentation_train_transform()
        train_dataset = MRBrainSPreprocessedDataset(split='train', transform=train_transform)
        
        validation_transform = segmentation_validation_transform()
        validation_dataset = MRBrainSPreprocessedDataset(split='validation', transform=validation_transform)

        pipeline = SegmentationOnlyPipeline(config=self)

        return pipeline.run(train_dataset, validation_dataset)

    def test(self):
        pass