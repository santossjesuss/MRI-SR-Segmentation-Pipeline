from torch.utils.data import random_split
from experiments.base_experiments import BaseExperiments
from experiments.experiment import Experiment
from configs.mslesseg_config import MSLesSeg
from datasets.mslesseg_dataset import MSLesSegDataset
from enums.resolution_enum import Resolution
from pipelines.super_resolution_pipeline import SuperResolutionPipeline
from pipelines.segmentation_pipeline import SegmentationPipeline

class MSLesSegExperiments(BaseExperiments):
    def __init__(self):
        super().__init__()
        self.config = MSLesSeg()

        complete_train_dataset = MSLesSegDataset(isTraining=True, view=self.config.view, scale_factor=self.config.scale_factor)
        train_size, validation_size = super()._get_train_validation_sizes(len(complete_train_dataset), self.config.train_perc_size)
        train_subset, validation_subset = random_split(complete_train_dataset, [train_size, validation_size])

        self.train_dataset = train_subset
        self.validation_dataset = validation_subset
        self.test_dataset = MSLesSegDataset(isTraining=False, view=self.config.view, scale_factor=self.config.scale_factor)

    def get_super_resolution(self):
        return Experiment(
            config=self.config,
            name=self.config.sr_saving_name,
            pipeline=SuperResolutionPipeline,
            training_dataset=self.train_dataset,
            validation_dataset=self.validation_dataset,
            test_dataset=self.test_dataset
        )
    
    def get_hr_segmentation(self):
        return Experiment(
            config=self.config,
            name=self.config.hr_seg_saving_name,
            pipeline=SegmentationPipeline,
            training_dataset=self.train_dataset,
            validation_dataset=self.validation_dataset,
            test_dataset=self.test_dataset,
            data_resolution=Resolution.HR
        )

    def get_lr_segmentation(self):
        return Experiment(
            config=self.config,
            name=self.config.lr_seg_saving_name,
            pipeline=SegmentationPipeline,
            training_dataset=self.train_dataset,
            validation_dataset=self.validation_dataset,
            test_dataset=self.test_dataset,
            data_resolution=Resolution.LR
        )

    def get_frozen_sr_frozen_seg(self):
        pass

    def get_frozen_sr_trainable_seg(self):
        pass

    def get_trainable_sr_frozen_seg(self):
        pass

    def get_joint_sr_seg(self):
        pass