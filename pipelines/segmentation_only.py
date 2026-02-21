import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from utils.gpu import enable_cuda
from training.unet_trainer import UnetTrainer
from utils.segmentation_metrics import SegmentationMetrics

class SegmentationOnlyPipeline:
    def __init__(self, config):
        self.config = config

    def run(self, train_dataset, validation_dataset):
        device = enable_cuda()
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=self.config.shuffle_data,
            num_workers=2
        )
        validation_loader = DataLoader(
            validation_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False,
            num_workers=2
        )

        model = smp.Unet(
            encoder_name=self.config.model_name,
            encoder_weights=None,
            in_channels=self.config.in_channels,
            classes=self.config.classes
        )
        validation_metrics = SegmentationMetrics(
            num_classes=self.config.classes, 
            ignore_index=self.config.ignore_index, 
            device=device
        )
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5
        )

        trainer = UnetTrainer(
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            criterion=self.config.criterion,
            validation_metrics=validation_metrics,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            saving_name=self.config.saving_name
        )
        
        return trainer.train(epochs=self.config.epochs)