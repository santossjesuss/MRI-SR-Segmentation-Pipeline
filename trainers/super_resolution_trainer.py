import torch
from tqdm import tqdm
from trainers.base_trainer import BaseTrainer
from utils.model_persistence import save_model_for_inference, load_model_for_inference

class SuperResolutionTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self, epochs):
        best_validation_score = float('-inf')

        for epoch in range(epochs):
            train_loss = self._train_epoch(epoch, epochs)
            validation_metrics = self._validate()

            validation_score = validation_metrics['psnr_mean']
            self.scheduler.step(validation_score)

            print(f'Epoch {epoch+1}/{epochs}')
            print(f'\tTrain Loss: {train_loss:.4f}')
            print(f'\tValidation Score: {validation_score:.4f} (PSNR)')
            print(f'\tValidation Metrics: {validation_metrics}')

            if validation_score > best_validation_score:
                best_validation_score = validation_score
                save_model_for_inference(self.model, self.saving_name)

        load_model_for_inference(self.saving_name)
        return self.model

    def _train_epoch(self, epoch, total_epochs):
        self.model.train()
        epoch_loss = 0

        progress_bar_description = f"Epoch {epoch+1}/{total_epochs}"
        progress_bar = tqdm(self.train_loader, desc=progress_bar_description)
        for lr_image, hr_image in progress_bar:
            lr_image = lr_image.to(self.device, dtype=torch.float32)
            hr_image = hr_image.to(self.device, dtype=torch.float32)

            self.optimizer.zero_grad(set_to_none=True)
            sr_image = self.model(lr_image)
            loss = self.criterion(sr_image, hr_image)

            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        return epoch_loss / len(self.train_loader)

    def _evaluate(self):
        self.model.eval()
        self.validation_metrics.reset()

        with torch.no_grad():
            for lr_image, hr_image in tqdm(self.validation_loader, desc='validating'):
                lr_image = lr_image.to(self.device, dtype=torch.float32)
                hr_image = hr_image.to(self.device, dtype=torch.float32)

                sr_image = self.model(lr_image)
                self.validation_metrics.update(sr_image, hr_image)

        return self.validation_metrics.compute()
    
    def _validate(self):
        return self._evaluate(self.validation_loader, description='Validating')
    
    def test(self, test_loader):
        return self._evaluate(test_loader, description='Testing')