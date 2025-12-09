import torch
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from comb.integration.hf.CombLlama import CombLlamaConfig, CombLlamaForConditionalGeneration

CONFIG_MAPPING = {
    'Llama': CombLlamaConfig
}

MODEL_CLASS_MAPPING = {
    'Llama': CombLlamaForConditionalGeneration
}

def get_config(model: str) -> PretrainedConfig:
    for model_type in CONFIG_MAPPING:
        if model_type in model:
            return CONFIG_MAPPING[model_type].from_pretrained(model)

    raise NotImplementedError(f"Model {model} is not implemented.")

def get_model(model: str) -> PreTrainedModel:
    for model_type in MODEL_CLASS_MAPPING:
        if model_type in model:
            model_class = MODEL_CLASS_MAPPING[model_type]
            return model_class.from_pretrained(
                model,
                torch_dtype=torch.bfloat16,
            )

    raise NotImplementedError(f"Model {model} is not implemented.")

def get_pic_shape(model: str) -> torch.Size:
    config = get_config(model)
    return torch.Size((
            len(config.cross_attention_layers),
            2,        # Key and value
            config.text_config.num_key_value_heads,
            config.text_config.head_dim
        ))