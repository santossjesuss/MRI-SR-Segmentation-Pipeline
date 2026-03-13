import os
from losses.dice_ce_combined_loss import DiceCECombinedLoss
from pipelines.base_pipeline import BasePipeline
from metrics.segmentation_metrics import SegmentationMetrics
from trainers.segmentation_trainer import SegmentationTrainer
from utils.model_persistence import load_model_for_inference

class SegmentationPipeline(BasePipeline):
    def __init__(self, config, saving_path):
        saving_path = os.path.join(config.saving_folder, saving_path)
        super().__init__(config, saving_path)

    def run(self, train_dataset, validation_dataset, data_resolution):
        train_loader = self._get_dataloader(train_dataset)
        validation_loader = self._get_dataloader(validation_dataset)

        model = self._init_unet()
        criterion = DiceCECombinedLoss(
            dice_weight=self.config.dice_weight, 
            cross_entropy_weight=self.config.cross_entropy_weight
        )
        validation_metrics = SegmentationMetrics(
            num_classes=self.config.seg_classes
        )
        optimizer = self._get_optimizer(model.parameters())
        scheduler = self._get_scheduler(optimizer)

        trainer = SegmentationTrainer(
            model=model,
            device=self.device,
            train_loader=train_loader,
            validation_loader=validation_loader,
            data_resolution=data_resolution,
            criterion=criterion,
            validation_metrics=validation_metrics,
            optimizer=optimizer,
            scheduler=scheduler,
            saving_name=self.saving_path
        )
        
        return trainer.train(epochs=self.config.epochs)
    
    def test(self, test_dataset, data_resolution):
        test_loader = self._get_dataloader(test_dataset)
        
        model = self._init_unet()
        validation_metrics = SegmentationMetrics(
            num_classes=self.config.seg_classes, 
            ignore_index=self.config.ignore_index
        )

        load_model_for_inference(model, self.config.seg_saving_name)

        trainer = SegmentationTrainer(
            model=model,
            device=self.device,
            validation_metrics=validation_metrics,
            data_resolution=data_resolution
        )

        return trainer.test(test_loader)