from pipelines.base_pipeline import BasePipeline
from trainers.segmentation_trainer import SegmentationTrainer
from utils.model_persistence import load_model_for_inference

class SegmentationPipeline(BasePipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, train_dataset, validation_dataset):
        train_loader = self._get_dataloader(train_dataset)
        validation_loader = self._get_dataloader(validation_dataset)

        model = self._init_unet()
        criterion = self._get_seg_loss()
        validation_metrics = self._get_seg_validation_metrics()
        optimizer = self._get_optimizer(model.parameters())
        scheduler = self._get_scheduler(optimizer)
        logger = self._get_logger()

        trainer = SegmentationTrainer(
            model=model,
            device=self.device,
            train_loader=train_loader,
            validation_loader=validation_loader,
            data_resolution=self.data_resolution,
            criterion=criterion,
            validation_metrics=validation_metrics,
            optimizer=optimizer,
            scheduler=scheduler,
            saving_name=self.saving_path,
            logger=logger
        )
        
        return trainer.train(epochs=self.config.epochs)
    
    def test(self, test_dataset):
        test_loader = self._get_dataloader(test_dataset)
        
        model = self._init_unet()
        validation_metrics = self._get_seg_validation_metrics()

        load_model_for_inference(model, self.saving_path)

        trainer = SegmentationTrainer(
            model=model,
            device=self.device,
            validation_metrics=validation_metrics,
            data_resolution=self.data_resolution
        )

        return trainer.test(test_loader)