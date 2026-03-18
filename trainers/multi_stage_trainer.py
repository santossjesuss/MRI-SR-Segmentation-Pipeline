import torch
from tqdm import tqdm
from trainers.base_trainer import BaseTrainer

class MultiStageTrainer(BaseTrainer):
    def __init__(self, *args, use_combined_loss=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_combined_loss = use_combined_loss

    def _train_epoch(self, epoch, total_epochs):
        self.model.train()
        epoch_loss = 0

        progress_bar_description = f"Epoch {epoch+1}/{total_epochs}"
        progress_bar = tqdm(self.train_loader, desc=progress_bar_description)
        for batch in progress_bar:
            lr_image, hr_image, hr_masks = self._prepare_batch(batch)

            self.optimizer.zero_grad(set_to_none=True)
            pred_hr_masks_logits, pred_hr_image = self.model(lr_image)
            if self.use_combined_loss:
                loss = self.criterion(pred_hr_image, hr_image, pred_hr_masks_logits, hr_masks) # create combined loss
            else:
                loss = self.criterion(pred_hr_masks_logits, hr_masks)
            
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
                lr_image, hr_masks = self._prepare_batch(batch)

                pred_hr_masks_logits = self.model(lr_image)                    # (Batch, Classes, H, W)
                predicted_masks = torch.argmax(pred_hr_masks_logits, dim=1)    # (Batch, H, W)

                self.validation_metrics.update(predicted_masks, hr_masks)

        return self.validation_metrics.compute()
    
    def _prepare_batch(self, batch):
        hr_image, hr_masks, lr_image, _ = batch
        
        hr_image = hr_image.to(self.device, dtype=torch.float32)
        lr_image = lr_image.to(self.device, dtype=torch.float32)
        hr_masks = hr_masks.to(self.device, dtype=torch.long)

        return lr_image, hr_image, hr_masks

    def get_primary_metric_name(self):
        return "dice"   # might use other metric