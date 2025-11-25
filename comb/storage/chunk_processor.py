from huggingface_hub import hf_hub_download
import json
import os
from safetensors import safe_open
import torch
from typing import Union

from comb.config import get_config, get_model_class

class ChunkProcessor:
    def __init__(
        self,
        model: str,
        rank: int,
    ) -> None:
        self.device = torch.device(f"cuda:{rank}")
        self.load_weights(model)

    def load_weights(self, model: str) -> None:
        if os.path.exists(model):
            path_func = os.path.join
        else:
            path_func = hf_hub_download
        
        index_file = path_func(model, "model.safetensors.index.json")
        with open(index_file) as f:
            index = json.load(f)

        chunk_keys = {k: v for k, v in index["weight_map"].items()
                    if k.startswith("chunk_model.")}
        needed_shards = set(chunk_keys.values())
        local_shards = [path_func(model, shard) for shard in needed_shards]
        state_dict = {}
        for shard_file in local_shards:
            with safe_open(shard_file, framework="pt", device=self.device) as f:
                for k in f.keys():
                    if k in chunk_keys:
                        state_dict[k.lstrip("chunk_model.")] = f.get_tensor(k)

        config = get_config(model).from_pretrained(model)
        model_class = get_model_class(model, chunk_or_backbone=True)
        self.chunk_model = model_class(config=config)
        self.chunk_model.load_state_dict(state_dict, strict=True)

    def process(tokens: Union[torch.Tensor, list[int]]) -> torch.Tensor:
        tokens = torch.tensor(tokens, device=self.device)
        return self.chunk_model(tokens)