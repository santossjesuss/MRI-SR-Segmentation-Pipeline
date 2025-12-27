from .squeeze_and_excitation import SE_Block
from .efficient_channel_attention import ECA_Block

class AttentionFactory:
    @staticmethod
    def get_attention_module(attention_type, **kwargs):
        if attention_type == "se":
            return SE_Block(**kwargs)
        elif attention_type == "eca":
            return ECA_Block(**kwargs)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")