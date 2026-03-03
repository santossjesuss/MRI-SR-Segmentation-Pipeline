from abc import ABC, abstractmethod
import torch.nn as nn

class BaseMetrics(ABC, nn.Module):
    def __init__(self):
        super().__init__()
        self.metrics = nn.ModuleDict()

    @abstractmethod
    def update(self, predictions, targets):
        pass

    def compute(self):
        results = {}
        for name, metric in self.metrics.items():
            value = metric.compute()
            results[name] = value.item() if value.dim() == 0 else value.tolist()
    
        return results

    def reset(self):
        for metric in self.metrics.values():
            metric.reset()