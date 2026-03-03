import torch.nn as nn
from segmentation_models_pytorch.losses import DiceLoss

class DiceCECombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.5, cross_entropy_weight=0.5):
        super().__init__()
        self.cross_entropy_weight = cross_entropy_weight
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.dice_weight = dice_weight
        self.dice_loss = DiceLoss(mode='multiclass', from_logits=True)

    def forward(self, predicted_masks_logits, true_masks):
        ce_loss = self.cross_entropy_loss(predicted_masks_logits, true_masks)
        dice_loss = self.dice_loss(predicted_masks_logits, true_masks)

        combined_loss = self.cross_entropy_weight * ce_loss + self.dice_weight * dice_loss
        return combined_loss