from abc import ABC, abstractmethod
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from models.rcan.rcan import RCAN
from losses.dice_ce_combined_loss import DiceCECombinedLoss
from losses.sr_seg_combined_loss import SRSegCombinedLoss
from metrics.superres_metrics import SuperResolutionMetrics
from metrics.segmentation_metrics import SegmentationMetrics
from loggers.tensorboard_logger import TensorBoardLogger
from utils.gpu import enable_cuda

class BasePipeline(ABC):
    def __init__(self, config, experiment_name, data_resolution=None):
        self.config = config
        self.device = enable_cuda()
        self.experiment_name = experiment_name
        self.saving_path = os.path.join(self.config.folder_name, f'{experiment_name}.pth')
        self.data_resolution = data_resolution

    @abstractmethod
    def run(self, train_dataset, validation_dataset):
        pass

    @abstractmethod
    def test(self, test_dataset):
        pass

    def _get_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle_data,
            num_workers=self.config.num_workers
        )
    
    def _init_rcan(self):
        return RCAN(
            in_channels=self.config.in_channels,
            out_channels=self.config.sr_out_channels,
            num_rg=self.config.num_rg,
            num_rcab=self.config.num_rcab,
            channels=self.config.sr_inner_channels,
            upscale_factor=self.config.scale_factor
        )
    
    def _init_unet(self):
        return smp.Unet(
            encoder_name=self.config.seg_model_name,
            encoder_weights=self.config.seg_encoder_weights,
            in_channels=self.config.in_channels,
            classes=self.config.seg_classes
        )
    
    def _get_sr_loss(self):
        return nn.L1Loss()

    def _get_seg_loss(self):
        return DiceCECombinedLoss(
            dice_weight=self.config.dice_weight, 
            cross_entropy_weight=self.config.cross_entropy_weight
        )

    def _get_combined_sr_seg_loss(self):
        sr_loss = self._get_sr_loss()
        seg_loss = self._get_seg_loss()
        
        return SRSegCombinedLoss(
            sr_loss_fn=sr_loss,
            seg_loss_fn=seg_loss,
            sr_weight=self.config.sr_loss_weight,
            seg_weight=self.config.seg_loss_weight
        )

    def _get_sr_validation_metrics(self):
        return SuperResolutionMetrics()

    def _get_seg_validation_metrics(self):
        return SegmentationMetrics(
            num_classes=self.config.seg_classes
        )

    def _get_optimizer(self, model_params):
        return optim.Adam(
            model_params,
            lr=self.config.learning_rate
        )
    
    def _get_scheduler(self, optimizer):
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5
        )
    
    def _get_logger(self):
        log_path = os.path.join("logs", self.experiment_name)
        return TensorBoardLogger(log_dir=log_path)