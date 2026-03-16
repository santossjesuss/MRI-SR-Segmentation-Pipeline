import torch
import torch.nn.functional as F
from tqdm import tqdm
from trainers.base_trainer import BaseTrainer
from enums.resolution_enum import Resolution

class SegmentationTrainer(BaseTrainer):
    def __init__(self, data_resolution=Resolution.HR, **kwargs):
        super().__init__(**kwargs)
        self.data_resolution = data_resolution
    
    def _train_epoch(self, epoch, total_epochs):
        self.model.train()
        epoch_loss = 0

        progress_bar_description = f"Epoch {epoch+1}/{total_epochs}"
        progress_bar = tqdm(self.train_loader, desc=progress_bar_description)
        for batch in progress_bar:
            image, masks = self._prepare_batch(batch)

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
            for batch in tqdm(dataloader, desc=description):
                image, masks = self._prepare_batch(batch)

                predicted_masks_logits = self.model(image)                      # (Batch, Classes, H, W)
                predicted_masks = torch.argmax(predicted_masks_logits, dim=1)   # (Batch, H, W)
                
                self.validation_metrics.update(predicted_masks, masks)

        return self.validation_metrics.compute()
    
    def _prepare_batch(self, batch):
        hr_image, hr_masks, lr_image, lr_masks = batch
        if self.data_resolution == Resolution.HR:
            image = hr_image.to(self.device, dtype=torch.float32)
            masks = hr_masks.to(self.device, dtype=torch.long)
        else:
            image = lr_image.to(self.device, dtype=torch.float32)
            masks = lr_masks.to(self.device, dtype=torch.long)

        return image, masks

    def get_primary_metric_name(self):
        return "dice"