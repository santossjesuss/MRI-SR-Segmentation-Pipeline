import torch
from tqdm import tqdm
from training.base_trainer import BaseTrainer

class UnetTrainer(BaseTrainer):
    def __init__(self, model, train_loader, validation_loader, criterion, validation_metrics, optimizer, scheduler, device, saving_name):
        super().__init__()
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.criterion = criterion
        self.validation_metrics = validation_metrics
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.saving_name = saving_name

    def train(self, epochs):
        best_validation_score = float('-inf')

        for epoch in range(epochs):
            train_loss = self._train_epoch(epoch, epochs)
            validation_metrics = self._validate()

            validation_score = validation_metrics['dice_mean']
            self.scheduler.step(validation_score)

            print(f'Epoch {epoch+1}/{epochs}')
            print(f'\tTrain Loss: {train_loss:.4f}')
            print(f'\tValidation Score: {validation_score:.4f} (Dice)')
            print(f'\tValidation Metrics: {validation_metrics}')

            if validation_score > best_validation_score:
                best_validation_score = validation_score
                torch.save({
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'validation_score': validation_score
                }, self.saving_name)

        return self.model # maybe return the best model instead of the last one or remove this line
    
    def _train_epoch(self, epoch, total_epochs):
        self.model.train()
        epoch_loss = 0

        progress_bar_description = f"Epoch {epoch+1}/{total_epochs}"
        progress_bar = tqdm(self.train_loader, desc=progress_bar_description)
        for image, masks in progress_bar:
            image = image.to(self.device, dtype=torch.float32)
            masks = masks.to(self.device, dtype=torch.long)

            self.optimizer.zero_grad()
            predicted_masks_logits = self.model(image)
            loss = self.criterion(predicted_masks_logits, masks)
            
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        return epoch_loss / len(self.train_loader)

    def _validate(self):
        self.model.eval()
        self.validation_metrics.reset()

        with torch.no_grad():
            for image, masks in tqdm(self.validation_loader, desc="Validating"):
                image = image.to(self.device, dtype=torch.float32)
                masks = masks.to(self.device, dtype=torch.long)

                predicted_masks_logits = self.model(image)                      # (Batch, Classes, H, W)
                predicted_masks = torch.argmax(predicted_masks_logits, dim=1)   # (Batch, 1, H, W) <- each pixel has the number associated with the class

                self.validation_metrics.update(predicted_masks, masks.int())

        return self.validation_metrics.compute()

    # def _validate(self):
    #     self.model.eval()
    #     for metric in self.validation_metrics.values():
    #         metric.reset()

    #     with torch.no_grad():
    #         for image, masks in tqdm(self.validation_loader, desc="Validating"):
    #             image = image.to(self.device)
    #             masks = masks.to(self.device)

    #             predicted_masks_logits = self.model(image) 
    #             predicted_masks = torch.argmax(predicted_masks_logits, dim=1)

    #             for metric in self.validation_metrics.values():
    #                 metric.update(predicted_masks, masks)

    #     return {k: v.compute() for k, v in self.validation_metrics.items()}