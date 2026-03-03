import torch.nn as nn

class MultiStageModel(nn.Module):
    def __init__(self, model_stage_1, model_stage_2, freeze_stage_1=False, freeze_stage_2=False):
        super(MultiStageModel, self).__init__()
        self.freeze_stage_1 = freeze_stage_1
        self.freeze_stage_2 = freeze_stage_2
        self.model_stage_1 = model_stage_1
        self.model_stage_2 = model_stage_2

        self._set_grad(self.model_stage_1, not freeze_stage_1)
        self._set_grad(self.model_stage_2, not freeze_stage_2)

    def forward(self, x):
        x = self.model_stage_1(x)
        x = self.model_stage_2(x)
        return x

    def _set_grad(self, model, required_grad):
        for param in model.parameters():
            param.requires_grad = required_grad

    def is_freezed_stage(self, stage_num):
        if stage_num < 1 or stage_num > 2:
            raise RuntimeError('Passed inexistent stage.')
        
        return self.freeze_stage_1 if stage_num == 1 else self.freeze_stage_2