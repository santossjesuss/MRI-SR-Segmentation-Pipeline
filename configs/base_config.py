from dataclasses import dataclass

@dataclass
class BaseConfig:
    # Training config
    epochs: int = 50
    batch_size: int = 2
    learning_rate: float = 1e-4
    shuffle_data: bool = True
    num_workers: int = 0
    # num_workers: int = 2

    # SuperRes config
    num_rg: int = 2
    num_rcab: int = 4
    sr_inner_channels: int = 64

    # Segmentation config
    seg_model_name: str = 'resnet34'
    seg_encoder_weights: str = None

    # Losses config
    dice_weight: float = 0.5
    cross_entropy_weight: float = 0.5
    sr_loss_weight: float = 0.5
    seg_loss_weight: float = 0.5

    # Saving config
    folder_name: str = 'trained_models'
    sr_name: str = 'sr'
    hr_seg_name: str = 'hr_seg'
    lr_seg_name: str = 'lr_seg'
    frozen_sr_frozen_seg: str = 'frozen_sr_frozen_seg'
    frozen_sr_trainable_seg: str = 'frozen_sr_trainable_seg'
    trainable_sr_frozen_seg: str = 'trainable_sr_frozen_seg'
    joint_sr_seg_e2e: str = 'joint_sr_seg_e2e'
    joint_sr_seg_combined: str = 'joint_sr_seg_combined'