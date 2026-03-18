import torch.nn as nn

class SRSegCombinedLoss(nn.Module):
    def __init__(self, sr_loss_fn, seg_loss_fn, sr_weight=0.5, seg_weight=0.5):
        self.sr_loss_fn = sr_loss_fn
        self.seg_loss_fn = seg_loss_fn
        self.sr_weight = sr_weight
        self.seg_weight = seg_weight
        
        self._validate_params()

    def forward(self, pred_hr_img, hr_img, pred_hr_masks_logits, hr_masks):
        sr_loss = self.sr_loss_fn(pred_hr_img, hr_img)
        seg_loss = self.seg_loss_fn(pred_hr_masks_logits, hr_masks)

        return sr_loss * self.sr_weight + seg_loss * self.seg_weight
    
    def _validate_params(self):
        if self.sr_weight + self.seg_weight != 1.0:
            raise ValueError('Sum of SuperRes and Segmentation weights must be 1 when combining losses.')