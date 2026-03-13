from abc import ABC, abstractmethod
from enums.resolution_enum import Resolution

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
            saving_name=None
        ):
        super().__init__()
        self.device = device
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.validation_metrics = validation_metrics.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.saving_name = saving_name    

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def _train_epoch(self, epoch, total_epochs):
        pass

    @abstractmethod
    def _evaluate(self, dataloader, description):
        pass

    @abstractmethod
    def _validate(self):
        pass

    @abstractmethod
    def test(self, test_loader):
        pass