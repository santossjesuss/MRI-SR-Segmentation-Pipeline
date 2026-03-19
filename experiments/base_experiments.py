from abc import ABC
from torch.utils.data import random_split
from experiments.experiment import Experiment
from enums.resolution_enum import Resolution
from pipelines.super_resolution_pipeline import SuperResolutionPipeline
from pipelines.segmentation_pipeline import SegmentationPipeline
from pipelines.frozen_sr_frozen_seg_pipeline import FrozenSRFrozenSegPipeline
from pipelines.frozen_sr_trainable_seg_pipeline import FrozenSRTrainableSegPipeline
from pipelines.trainable_sr_frozen_seg_pipeline import TrainableSRFrozenSegPipeline
from pipelines.joint_sr_seg_e2e_pipeline import JointSRSegE2EPipeline
from pipelines.joint_sr_seg_combined_pipeline import JointSRSegCombinedPipeline

class BaseExperiments(ABC):
    def __init__(self, config, dataset):
        super().__init__()
        self.config = config

        complete_train_dataset = dataset(
            is_training=True, 
            view=self.config.view, 
            scale_factor=self.config.scale_factor
        )
        train_size, validation_size = self._get_train_validation_sizes(
            train_dataset_size=len(complete_train_dataset), 
            train_perc=self.config.train_perc_size
        )
        train_subset, validation_subset = random_split(
            dataset=complete_train_dataset, 
            lengths=[train_size, validation_size]
        )

        self.train_dataset = train_subset
        self.validation_dataset = validation_subset
        self.test_dataset = dataset(
            is_training=False, 
            view=self.config.view, 
            scale_factor=self.config.scale_factor
        )
    
    def _get_train_validation_sizes(self, train_dataset_size, train_perc):
        train_size = int(train_perc * train_dataset_size)
        validation_size = train_dataset_size - train_size

        return train_size, validation_size
    
    def _create_experiment(self, name, pipeline, data_resolution=None):
        return Experiment(
            config=self.config,
            name=name,
            pipeline=pipeline,
            training_dataset=self.train_dataset,
            validation_dataset=self.validation_dataset,
            test_dataset=self.test_dataset,
            data_resolution=data_resolution
        )

    def get_super_resolution(self):
        return self._create_experiment(
            name=self.config.sr_name, 
            pipeline=SuperResolutionPipeline
        )

    def get_hr_segmentation(self):
        return self._create_experiment(
            name=self.config.hr_seg_name, 
            pipeline=SegmentationPipeline,
            data_resolution=Resolution.HR
        )

    def get_lr_segmentation(self):
        return self._create_experiment(
            name=self.config.lr_seg_name, 
            pipeline=SegmentationPipeline,
            data_resolution=Resolution.LR
        )

    def get_frozen_sr_frozen_seg(self):
        return self._create_experiment(
            name=self.config.frozen_sr_frozen_seg, 
            pipeline=FrozenSRFrozenSegPipeline
        )

    def get_frozen_sr_trainable_seg(self):
        return self._create_experiment(
            name=self.config.frozen_sr_trainable_seg, 
            pipeline=FrozenSRTrainableSegPipeline
        )

    def get_trainable_sr_frozen_seg(self):
        return self._create_experiment(
            name=self.config.trainable_sr_frozen_seg, 
            pipeline=TrainableSRFrozenSegPipeline
        )

    # This pipeline uses Segmentation Loss to learn
    def get_joint_sr_seg_e2e(self):
        return self._create_experiment(
            name=self.config.joint_sr_seg_e2e, 
            pipeline=JointSRSegE2EPipeline
        )

    # This pipeline uses both SuperRes and Segmentation Losses to learn
    def get_joint_sr_seg_combined(self):
        return self._create_experiment(
            name=self.config.joint_sr_seg_combined, 
            pipeline=JointSRSegCombinedPipeline
        )