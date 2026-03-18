from torch.utils.data import random_split
from experiments.base_experiments import BaseExperiments
from experiments.experiment import Experiment
from configs.mslesseg_config import MSLesSegConfig
from datasets.mslesseg_dataset import MSLesSegDataset
from enums.resolution_enum import Resolution
from pipelines.super_resolution_pipeline import SuperResolutionPipeline
from pipelines.segmentation_pipeline import SegmentationPipeline
from pipelines.frozen_sr_frozen_seg_pipeline import FrozenSRFrozenSegPipeline
from pipelines.frozen_sr_trainable_seg_pipeline import FrozenSRTrainableSegPipeline
from pipelines.trainable_sr_frozen_seg_pipeline import TrainableSRFrozenSegPipeline
from pipelines.joint_sr_seg_e2e_pipeline import JointSRSegE2EPipeline
from pipelines.joint_sr_seg_combined_pipeline import JointSRSegCombinedPipeline

class MSLesSegExperiments(BaseExperiments):
    def __init__(self):
        super().__init__()
        self.config = MSLesSegConfig()

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
        return Experiment(
            config=self.config,
            name=self.config.frozen_sr_frozen_seg,
            pipeline=FrozenSRFrozenSegPipeline,
            training_dataset=self.train_dataset,
            validation_dataset=self.validation_dataset,
            test_dataset=self.test_dataset
        )

    def get_frozen_sr_trainable_seg(self):
        return Experiment(
            config=self.config,
            name=self.config.frozen_sr_trainable_seg,
            pipeline=FrozenSRTrainableSegPipeline,
            training_dataset=self.train_dataset,
            validation_dataset=self.validation_dataset,
            test_dataset=self.test_dataset
        )

    def get_trainable_sr_frozen_seg(self):
        return Experiment(
            config=self.config,
            name=self.config.trainable_sr_frozen_seg,
            pipeline=TrainableSRFrozenSegPipeline,
            training_dataset=self.train_dataset,
            validation_dataset=self.validation_dataset,
            test_dataset=self.test_dataset
        )

    # This pipeline uses Segmentation Loss to learn
    def get_joint_sr_seg_e2e(self):
        return Experiment(
            config=self.config,
            name=self.config.joint_sr_seg_e2e,
            pipeline=JointSRSegE2EPipeline,
            training_dataset=self.train_dataset,
            validation_dataset=self.validation_dataset,
            test_dataset=self.test_dataset
        )
    
    # This pipeline uses both SuperRes and Segmentation Losses to learn
    def get_joint_sr_seg_combined(self):
        return Experiment(
            config=self.config,
            name=self.config.joint_sr_seg_combined,
            pipeline=JointSRSegCombinedPipeline,
            training_dataset=self.train_dataset,
            validation_dataset=self.validation_dataset,
            test_dataset=self.test_dataset
        )