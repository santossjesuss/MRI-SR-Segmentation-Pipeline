from pipelines.base_pipeline import BasePipeline
from trainers.super_resolution_trainer import SuperResolutionTrainer
from utils.model_persistence import load_model_for_inference

class SuperResolutionPipeline(BasePipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, train_dataset, validation_dataset):
        train_loader = self._get_dataloader(train_dataset)
        validation_loader = self._get_dataloader(validation_dataset)

        model = self._init_rcan()
        criterion = self._get_sr_loss()
        validation_metrics = self._get_sr_validation_metrics()
        optimizer = self._get_optimizer(model.parameters())
        scheduler = self._get_scheduler(optimizer)
        logger = self._get_logger()

        trainer = SuperResolutionTrainer(
            model=model,
            device=self.device,
            train_loader=train_loader,
            validation_loader=validation_loader,
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

        model = self._init_rcan()
        validation_metrics = self._get_sr_validation_metrics()

        load_model_for_inference(model, self.saving_path)

        trainer = SuperResolutionTrainer(
            model=model,
            device=self.device,
            validation_metrics=validation_metrics
        )

        return trainer.test(test_loader)