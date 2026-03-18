from abc import ABC, abstractmethod
from utils.model_persistence import save_model_for_inference, load_model_for_inference

class BaseTrainer(ABC):
    def __init__(
            self, 
            model, 
            device, 
            train_loader=None, 
            validation_loader=None, 
            criterion=None, 
            validation_metrics=None, 
            optimizer=None, 
            scheduler=None, 
            saving_name=None,
            logger=None
        ):
        super().__init__()
        self.device = device
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.model = model.to(device)
        self.criterion = criterion.to(device) if criterion else None
        self.validation_metrics = validation_metrics.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.saving_name = saving_name
        self.logger = logger

    def train(self, epochs):
        best_validation_score = float('-inf')

        for epoch in range(epochs):
            train_loss = self._train_epoch(epoch, epochs)
            validation_metrics = self._validate()

            validation_score = validation_metrics[self.get_primary_metric_name()]
            self.scheduler.step(validation_score)

            if self.logger:
                self.logger.log_metric("Loss", train_loss, epoch, phase="Training")
                self.logger.log_metrics(validation_metrics, epoch, phase="Validation")

            if validation_score > best_validation_score:
                best_validation_score = validation_score
                save_model_for_inference(self.model, self.saving_name)
            
            print(f'Epoch {epoch+1}/{epochs}')
            print(f'\tTrain Loss: {train_loss:.4f}')
            print(f'\tValidation Metrics: {validation_metrics}')
            # print(f'\tValidation Main Score: {self.get_primary_metric_name()}')

        load_model_for_inference(self.saving_name)
        return self.model

    @abstractmethod
    def _train_epoch(self, epoch, total_epochs):
        pass

    @abstractmethod
    def _evaluate(self, dataloader, description):
        pass

    def _validate(self):
        return self._evaluate(self.validation_loader, description='Validating')
    
    def test(self, test_loader):
        return self._evaluate(test_loader, description='Testing')
    
    @abstractmethod
    def _prepare_batch(self, batch):
        pass

    @abstractmethod
    def get_primary_metric_name(self):
        pass