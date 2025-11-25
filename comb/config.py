import torch
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from comb.integration.hf.CombLlama import (CombLlamaConfig, CombLlamaChunkModel,
                                            CombLlamaForConditionalGeneration)

CONFIG_MAPPING = {
    'Llama': CombLlamaConfig
}

CHUNK_MODEL_MAPPING = {
    'Llama': CombLlamaChunkModel
}

BACKBONE_MODEL_MAPPING = {
    'Llama': CombLlamaForConditionalGeneration
}

def get_config(model: str) -> PretrainedConfig:
    for model_type in CONFIG_MAPPING:
        if model_type in model:
            return CONFIG_MAPPING[model_type].from_pretrained(model)

    raise NotImplementedError(f"Model {model} is not implemented.")

def get_model_class(model: str, chunk_or_backbone: bool) -> PreTrainedModel:
    mapping = CHUNK_MODEL_MAPPING if chunk_or_backbone else BACKBONE_MODEL_MAPPING
    for model_type in mapping:
        if model_type in model:
            return mapping[model_type]

    raise NotImplementedError(f"Model {model} is not implemented.")

def get_pic_shape(model: str) -> torch.Size:
    config = get_config(model)
    return torch.Size((
            len(config.cross_attention_layers),
            2,        # Key and value
            config.text_config.num_key_value_heads,
            config.text_config.head_dim
        ))