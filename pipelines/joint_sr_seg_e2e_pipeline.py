from pipelines.base_pipeline import BasePipeline
from models.multi_stage_model import MultiStageModel
from trainers.multi_stage_trainer import MultiStageTrainer
from utils.model_persistence import load_model_for_inference

class JointSRSegE2EPipeline(BasePipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, train_dataset, validation_dataset):
        train_loader = self._get_dataloader(train_dataset)
        validation_loader = self._get_dataloader(validation_dataset)
    
        sr_model = self._init_rcan()
        seg_model = self._init_unet()
        criterion = self._get_seg_loss()
        validation_metrics = self._get_seg_validation_metrics()

        joint_sr_seg_e2e_model = MultiStageModel(
            model_stage_1=sr_model,
            model_stage_2=seg_model,
            freeze_stage_1=False,
            freeze_stage_2=False
        )

        optimizer = self._get_optimizer(joint_sr_seg_e2e_model.parameters())
        scheduler = self._get_scheduler(optimizer)

        trainer = MultiStageTrainer(
            model=joint_sr_seg_e2e_model,
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

        joint_sr_seg_e2e_model = MultiStageModel(
            sr_model, 
            seg_model, 
            freeze_stage_1=True, 
            freeze_stage_2=True
        )
        
        load_model_for_inference(model=joint_sr_seg_e2e_model, saving_name=self.saving_path)

        trainer = MultiStageTrainer(
            model=joint_sr_seg_e2e_model,
            device=self.device,
            validation_metrics=validation_metrics
        )

        return trainer.test(test_loader)