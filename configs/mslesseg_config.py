from dataclasses import dataclass
from configs.base_config import BaseConfig

@dataclass
class MSLesSeg(BaseConfig):
    scale_factor: float = 2

    seg_model_name: str = 'resnet34'
    seg_encoder_weights: str = None
    in_channels: int = 1
    seg_classes: int = 2
    
    sr_saving_name: str = 'mslesseg_sr'
    seg_saving_name: str = 'mslesseg_seg'
    
    train_perc_size: float = 0.8
    view: str = 'axial'

    frozen_sr_frozen_seg: str = 'mslesseg_frozen_sr_frozen_seg (baseline)'
    frozen_sr_trainable_seg: str = 'mslesseg_frozen_sr_trainable_seg'   # seg model trained by freezing sr
    trainable_sr_frozen_seg: str = 'mslesseg_trainable_sr_frozen_seg'
    joint_sr_seg: str = 'mslesseg_joint_sr_seg'