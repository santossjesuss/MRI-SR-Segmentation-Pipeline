from dataclasses import dataclass

@dataclass
class BaseConfig:
    epochs: int = 50
    batch_size: int = 2
    learning_rate: float = 1e-4
    ignore_index: int = 0
    include_background: bool = False
    shuffle_data: bool = True
    # num_workers: int = 2
    num_workers: int = 0

    dice_weight: float = 0.5
    cross_entropy_weight: float = 0.5

    saving_folder: str = 'trained_models'