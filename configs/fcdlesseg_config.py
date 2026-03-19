from dataclasses import dataclass
from configs.base_config import BaseConfig

@dataclass
class FCDLesSegConfig(BaseConfig):
    # Common SuperRes and Seg config
    in_channels: int = 1

    # SuperRes config
    sr_out_channels: int = 1
    scale_factor: float = 2

    # Segmentation config
    seg_classes: int = 2
    
    # Dataset config
    train_perc_size: float = 0.8
    view: str = 'axial'

    # Saving config
    dataset_name = "fcdlesseg"
    sr_name: str = f'{dataset_name}_{BaseConfig.sr_name}'
    hr_seg_name: str = f'{dataset_name}_{BaseConfig.hr_seg_name}'
    lr_seg_name: str = f'{dataset_name}_{BaseConfig.lr_seg_name}'
    frozen_sr_frozen_seg: str = f'{dataset_name}_{BaseConfig.frozen_sr_frozen_seg}'
    frozen_sr_trainable_seg: str = f'{dataset_name}_{BaseConfig.frozen_sr_trainable_seg}'
    trainable_sr_frozen_seg: str = f'{dataset_name}_{BaseConfig.trainable_sr_frozen_seg}'
    joint_sr_seg_e2e: str = f'{dataset_name}_{BaseConfig.joint_sr_seg_e2e}'
    joint_sr_seg_combined: str = f'{dataset_name}_{BaseConfig.joint_sr_seg_combined}'