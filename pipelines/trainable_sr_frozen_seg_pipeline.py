import os
from pipelines.base_pipeline import BasePipeline
from models.multi_stage_model import MultiStageModel
from trainers.multi_stage_trainer import MultiStageTrainer
from utils.model_persistence import load_model_for_inference

class TrainableSRFrozenSegPipeline(BasePipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, train_dataset, validation_dataset):
        train_loader = self._get_dataloader(train_dataset)
        validation_loader = self._get_dataloader(validation_dataset)

        sr_model = self._init_rcan()
        seg_model = self._init_unet()
        criterion = self._get_seg_loss()
        validation_metrics = self._get_seg_validation_metrics()

        seg_path = os.path.join(self.config.saving_folder, f'{self.config.hr_seg_name}.pth')
        load_model_for_inference(model=seg_model, saving_name=seg_path)

        trainable_sr_frozen_seg_model = MultiStageModel(
            sr_model, 
            seg_model, 
            freeze_stage_1=False, 
            freeze_stage_2=True
        )
        optimizer = self._get_optimizer(trainable_sr_frozen_seg_model.parameters())
        scheduler = self._get_scheduler(optimizer)

        trainer = MultiStageTrainer(
            model=trainable_sr_frozen_seg_model,
            device=self.device,
            train_loader=train_loader,
            validation_loader=validation_loader,
            criterion=criterion,
            validation_metrics=validation_metrics,
            optimizer=optimizer,
            scheduler=scheduler,
            saving_name=self.saving_path
        )

        return trainer.train(epochs=self.config.epochs)
    
    def test(self, test_dataset):
        test_loader = self._get_dataloader(test_dataset)

        sr_model = self._init_rcan()
        seg_model = self._init_unet()
        validation_metrics = self._get_seg_validation_metrics()

        trainable_sr_frozen_seg_model = MultiStageModel(
            sr_model, 
            seg_model, 
            freeze_stage_1=True, 
            freeze_stage_2=True
        )

        load_model_for_inference(model=trainable_sr_frozen_seg_model, saving_name=self.saving_path)

        trainer = MultiStageTrainer(
            model=trainable_sr_frozen_seg_model,
            device=self.device,
            validation_metrics=validation_metrics
        )

        return trainer.test(test_loader)