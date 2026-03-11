from dataclasses import dataclass

@dataclass
class MRBrainSConfig:
    sr_out_channels: int = 3
    num_rg: int = 2
    num_rcab: int = 4
    sr_inner_channels: int = 60
    
    seg_model_name: str = 'resnet34'
    seg_encoder_weights: str = None
    in_channels:int = 3    # Modalities (T1, T1_IR, T2_FLAIR)
    # seg_classes: int = 9
    seg_classes: int = 3

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
    sr_saving_name: str = 'mrbrains_sr'
    seg_saving_name: str = 'mrbrains_seg'
    
    frozen_sr_frozen_seg: str = 'mrbrains_frozen_sr_frozen_seg (baseline)'
    frozen_sr_trainable_seg: str = 'mrbrains_frozen_sr_trainable_seg'   # seg model trained by freezing sr
    trainable_sr_frozen_seg: str = 'mrbrains_trainable_sr_frozen_seg'
    joint_sr_seg: str = 'mrbrains_joint_sr_seg'

    frozen_seg_frozen_sr: str = 'mrbrains_frozen_seg_frozen_sr'
    frozen_seg_trainable_sr: str = 'mrbrains_frozen_seg_trainable_sr'
    trainable_seg_frozen_sr: str = 'mrbrains_trainable_seg_frozen_sr'
    joint_seg_sr: str = 'mrbrains_joint_seg_sr'