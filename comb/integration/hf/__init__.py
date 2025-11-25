from transformers import AutoModel
from typing import Any

from comb.config import get_config, get_model_class

class HFEngine:
    def __init__(
        self,
        model_name,
        rank,
        **kwargs,
    ) -> None:
        self.device = torch.device(f"cuda:{rank}")
        if 'Comb' not in model_name:
            model = AutoModel.from_pretrained(model_name)
            self.model = model.to(device=torch.device(f'cuda:{rank}'))
        else:
            self.load_weights(model_name)

    def generate(
        self,
        prompts: dict[str, Any],
        **kwargs,
    ) -> None:
        return self.model.generate(**prompts, **kwargs)

    def load_weights(self, model: str) -> None:
        if os.path.exists(model):
            path_func = os.path.join
        else:
            path_func = hf_hub_download
        
        index_file = path_func(model, "model.safetensors.index.json")
        with open(index_file) as f:
            index = json.load(f)

        keys = {k: v for k, v in index["weight_map"].items()
                    if not k.startswith("chunk_model.")}
        needed_shards = set(keys.values())
        local_shards = [path_func(model, shard) for shard in needed_shards]
        state_dict = {}
        for shard_file in local_shards:
            with safe_open(shard_file, framework="pt", device=self.device) as f:
                for k in f.keys():
                    if k in keys:
                        state_dict[k] = f.get_tensor(k)

        config = get_config(model).from_pretrained(model)
        model_class = get_model_class(model, chunk_or_backbone=False)
        self.model = model_class(config=config)
        self.model.load_state_dict(state_dict, strict=True)