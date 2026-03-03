from configs.mrbrains_config import MRBrainSConfig
from experiments.base_experiments import BaseExperiments
from experiments.experiment import Experiment
from pipelines.super_resolution_pipeline import SuperResolutionPipeline
from pipelines.segmentation_pipeline import SegmentationPipeline
from pipelines.frozen_sr_frozen_seg_pipeline import FrozenSRFrozenSegPipeline
from pipelines.frozen_sr_trainable_seg_pipeline import FrozenSRTrainableSegPipeline
from pipelines.joint_sr_seg_pipeline import JointSRSegPipeline
from transforms.segmentation_transforms import segmentation_train_transform, segmentation_validation_transform
from datasets.mrbrains_preprocessed_dataset import MRBrainSPreprocessedDataset

class MRBRainSExperiments(BaseExperiments):
    def __init__(self):
        self.config = MRBrainSConfig()
        train_transform = segmentation_train_transform()
        validation_transform = segmentation_validation_transform()
        
        self.train_dataset = MRBrainSPreprocessedDataset(split='train', transform=train_transform)
        self.validation_dataset = MRBrainSPreprocessedDataset(split='validation', transform=validation_transform)
        self.test_dataset = MRBrainSPreprocessedDataset(split='test', transform=validation_transform) # por hacer

    def get_super_resolution(self):
        return Experiment(
            name=self.config.sr_saving_name,
            pipeline=SuperResolutionPipeline,
            config=self.config,
            training_dataset=self.train_dataset,
            validation_dataset=self.validation_dataset,
            test_dataset=self.test_dataset
        )
    
    def get_segmentation(self):
        return Experiment(
            name=self.config.seg_saving_name,
            pipeline=SegmentationPipeline,
            config=self.config,
            training_dataset=self.train_dataset,
            validation_dataset=self.validation_dataset,
            test_dataset=self.test_dataset
        )
    
    def get_frozen_sr_frozen_seg(self):
        return Experiment(
            name=self.config.frozen_sr_frozen_seg,
            pipeline=FrozenSRFrozenSegPipeline,
            config=self.config,
            training_dataset=self.train_dataset,
            validation_dataset=self.validation_dataset,
            test_dataset=self.test_dataset
        )

    def get_frozen_sr_trainable_seg(self):
        return Experiment(
            name=self.config.frozen_sr_trainable_seg,
            pipeline=FrozenSRTrainableSegPipeline,
            config=self.config,
            training_dataset=self.train_dataset,
            validation_dataset=self.validation_dataset,
            test_dataset=self.test_dataset
        )
    
    def get_trainable_sr_frozen_seg(self):
        return Experiment(
            name=self.config.trainable_sr_frozen_seg,
            pipeline=None,
            config=self.config,
            training_dataset=self.train_dataset,
            validation_dataset=self.validation_dataset,
            test_dataset=self.test_dataset
        )
    
    def get_joint_sr_seg(self):
        return Experiment(
            name=self.config.joint_sr_seg,
            pipeline=JointSRSegPipeline,
            config=self.config,
            training_dataset=self.train_dataset,
            validation_dataset=self.validation_dataset,
            test_dataset=self.test_dataset
        )
    
    def get_frozen_seg_frozen_sr(self):
        return Experiment(
            name=self.config.frozen_seg_frozen_sr,
            pipeline=None,
            config=self.config,
            training_dataset=self.train_dataset,
            validation_dataset=self.validation_dataset,
            test_dataset=self.test_dataset
        )

    def get_frozen_seg_trainable_sr(self):
        return Experiment(
            name='mrbrains_frozen_seg_trainable_sr',
            pipeline=None,
            config=self.config,
            training_dataset=self.train_dataset,
            validation_dataset=self.validation_dataset,
            test_dataset=self.test_dataset
        )

    def get_trainable_seg_frozen_sr(self):
        return Experiment(
            name='mrbrains_trainable_seg_frozen_sr',
            pipeline=None,
            config=self.config,
            training_dataset=self.train_dataset,
            validation_dataset=self.validation_dataset,
            test_dataset=self.test_dataset
        )
    
    def get_joint_seg_sr(self):
        pass