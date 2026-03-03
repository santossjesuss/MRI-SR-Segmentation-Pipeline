import torch
import torch.nn.functional as F
from tqdm import tqdm
from trainers.base_trainer import BaseTrainer
from utils.model_persistence import save_model_for_inference, load_model_for_inference

class SegmentationTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self, epochs):
        best_validation_score = float('-inf')

        for epoch in range(epochs):
            train_loss = self._train_epoch(epoch, epochs)
            validation_metrics = self._validate()

            validation_score = validation_metrics['dice']
            self.scheduler.step(validation_score)

            print(f'Epoch {epoch+1}/{epochs}')
            print(f'\tTrain Loss: {train_loss:.4f}')
            print(f'\tValidation Score: {validation_score:.4f} (Dice)')
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
        for image, masks in progress_bar:
            image = image.to(self.device, dtype=torch.float32)
            masks = masks.to(self.device, dtype=torch.long)

            self.optimizer.zero_grad(set_to_none=True)
            predicted_masks_logits = self.model(image)
            loss = self.criterion(predicted_masks_logits, masks)
            
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        return epoch_loss / len(self.train_loader)

    def _evaluate(self, dataloader, description):
        self.model.eval()
        self.validation_metrics.reset()

        with torch.no_grad():
            for image, masks in tqdm(dataloader, desc=description):
                image = image.to(self.device, dtype=torch.float32)
                masks = masks.to(self.device, dtype=torch.long)

                predicted_masks_logits = self.model(image)                      # (Batch, Classes, H, W)
                one_hot_masks = F.one_hot(masks, num_classes=9).permute(0, 3, 1, 2)
                self.validation_metrics.update(predicted_masks_logits, one_hot_masks)
                # predicted_masks = torch.argmax(predicted_masks_logits, dim=1)
                # self.validation_metrics.update(predicted_masks, masks)

        return self.validation_metrics.compute()

    def _validate(self):
        return self._evaluate(self.validation_loader, description='Validating')
    
    def test(self, test_loader):
        return self._evaluate(test_loader, description='Testing')