from pipelines.base_pipeline import BasePipeline
from models.multi_stage_model import MultiStageModel
from trainers.multi_stage_trainer import MultiStageTrainer
from utils.model_persistence import load_model_for_inference

class FrozenSRFrozenSegPipeline(BasePipeline):
    def __init__(self, config):
        super().__init__(config)

    def run(self, train_dataset, validation_dataset):
        print('This pipeline does not support training.')
        print('It has already both models trained.')

    def test(self, test_dataset):
        test_loader = self._get_dataloader(test_dataset)

        sr_model = self._init_rcan()
        seg_model = self._init_unet()
        validation_metrics = None

        load_model_for_inference(sr_model, self.config.sr_saving_name)
        load_model_for_inference(seg_model, self.config.seg_saving_name)

        sr_seg_model = MultiStageModel(
            sr_model, 
            seg_model, 
            freeze_stage_1=True, 
            freeze_stage_2=True
        )

        trainer = MultiStageTrainer(
            model=sr_seg_model,
            device=self.device,
            validation_metrics=validation_metrics
        )

        return trainer.test(test_loader)