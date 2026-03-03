import os
from pipelines.base_pipeline import BasePipeline
from models.multi_stage_model import MultiStageModel
from trainers.multi_stage_trainer import MultiStageTrainer
from utils.model_persistence import load_model_for_inference

class FrozenSRTrainableSegPipeline(BasePipeline):
    def __init__(self, config):
        self.config.frozen_sr_trainable_seg = os.path.join(self.config.saving_folder, self.config.frozen_sr_trainable_seg)
        super().__init__(config=config)

    def run(self, train_dataset, validation_dataset):
        train_loader = self._get_dataloader(train_dataset)
        validation_loader = self._get_dataloader(validation_dataset)

        sr_model = self._init_rcan()
        seg_model = self._init_unet()
        criterion = None
        validation_metrics = None

        load_model_for_inference(sr_model, self.config.sr_saving_name)

        frozen_sr_trainable_seg_model = MultiStageModel(
            sr_model, 
            seg_model, 
            freeze_stage_1=True, 
            freeze_stage_2=False
        )
        optimizer = self._get_optimizer(frozen_sr_trainable_seg_model.parameters())
        scheduler = self._get_scheduler(optimizer)

        trainer = MultiStageTrainer(
            model=frozen_sr_trainable_seg_model,
            device=self.device,
            train_loader=train_loader,
            validation_loader=validation_loader,
            criterion=criterion,
            validation_metrics=validation_metrics,
            optimizer=optimizer,
            scheduler=scheduler,
            saving_name=self.config.frozen_sr_trainable_seg
        )

        return trainer.train(epochs=self.config.epochs)

    def test(self, test_dataset):
        test_loader = self._get_dataloader(test_dataset)

        sr_model = self._init_rcan()
        seg_model = self._init_unet()
        validation_metrics = None

        frozen_sr_trainable_seg_model = MultiStageModel(
            sr_model, 
            seg_model, 
            freeze_stage_1=True, 
            freeze_stage_2=True
        )
        
        load_model_for_inference(model=frozen_sr_trainable_seg_model, saving_name=self.config.frozen_sr_trainable_seg)

        trainer = MultiStageTrainer(
            model=frozen_sr_trainable_seg_model,
            device=self.device,
            validation_metrics=validation_metrics
        )

        return trainer.test(test_loader)