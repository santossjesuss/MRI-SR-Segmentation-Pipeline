import torch
from tqdm import tqdm
from trainers.base_trainer import BaseTrainer
from utils.model_persistence import save_model_for_inference, load_model_for_inference

class MultiStageTrainer(BaseTrainer):
    def __int__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self, epochs):
        best_validation_score = float('-inf')

        for epoch in range(epochs):
            train_loss = self._train_epoch(epoch=epoch, total_epochs=epochs)
            validation_metrics = self._validate()

            validation_score = validation_metrics['dice_mean'] # might use other metric
            self.scheduler.step(validation_score)

            print(f'Epoch {epoch+1}/{epochs}')
            print(f'\tTrain Loss {train_loss:.4f}')
            print(f'\tValidation Score: {validation_score:.4f} (Dice)') # might use other metric
            print(f'\tValidation Metrics: {validation_metrics}')

            if validation_score > best_validation_score:
                best_validation_score = validation_score
                save_model_for_inference(model=self.model, saving_name=self.saving_name)

            load_model_for_inference(model=self.model, saving_name=self.saving_name, device=self.device) # device?
            return self.model

    def _train_epoch(self, epoch, total_epochs):
        self.model.train()
        epoch_loss = 0

        progress_bar_description = f"Epoch {epoch+1}/{total_epochs}"
        progress_bar = tqdm(self.train_loader, desc=progress_bar_description)
        for lr_image, hr_masks in progress_bar:
            lr_image = lr_image.to(self.device, dtype=torch.float32)
            hr_masks = hr_masks.to(self.device, dtype=torch.long)

            self.optimizer.zero_grad(set_to_none=True)
            predicted_hr_masks_logits = self.model(lr_image)
            loss = self.criterion(predicted_hr_masks_logits, hr_masks)
            
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        return epoch_loss / len(self.train_loader)

    def _evaluate(self, dataloader, description):
        self.model.eval()
        self.validation_metrics.reset()

        with torch.no_grad():
            for lr_image, hr_masks in tqdm(dataloader, desc=description):
                lr_image = lr_image.to(self.device, dtype=torch.float32)
                hr_masks = hr_masks.to(self.device, dtype=torch.long)

                predicted_hr_masks_logits = self.model(lr_image)                      # (Batch, Classes, H, W)
                self.validation_metrics.update(predicted_hr_masks_logits, hr_masks)

        return self.validation_metrics.compute()

    def _validate(self):
        return self._evaluate(self.validation_loader, description='Validating')
    
    def test(self, test_loader):
        return self._evaluate(test_loader, description='Testing')