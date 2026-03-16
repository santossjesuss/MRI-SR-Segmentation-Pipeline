import torch
from tqdm import tqdm
from trainers.base_trainer import BaseTrainer

class SuperResolutionTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _train_epoch(self, epoch, total_epochs):
        self.model.train()
        epoch_loss = 0

        progress_bar_description = f"Epoch {epoch+1}/{total_epochs}"
        progress_bar = tqdm(self.train_loader, desc=progress_bar_description)
        for batch in progress_bar:
            lr_image, hr_image = self._prepare_batch(batch)

            self.optimizer.zero_grad(set_to_none=True)
            sr_image = self.model(lr_image)
            loss = self.criterion(sr_image, hr_image)

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
                lr_image, hr_image = self._prepare_batch(batch)

                sr_image = self.model(lr_image)
                self.validation_metrics.update(sr_image, hr_image)

        return self.validation_metrics.compute()
    
    def _prepare_batch(self, batch):
        hr_image, _, lr_image, _ = batch

        lr_image = lr_image.to(self.device, dtype=torch.float32)
        hr_image = hr_image.to(self.device, dtype=torch.float32)

        return lr_image, hr_image

    def get_primary_metric_name(self):
        return "psnr"