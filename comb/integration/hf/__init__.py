import logging
import torch
from transformers import AutoModel
from typing import Any

from comb.config import get_model

logger = logging.getLogger(__name__)
class HFEngine:
    def __init__(
        self,
        model_name,
        rank,
        **kwargs,
    ) -> None:
        self.device = f"cuda:{rank}"
        if 'Comb' not in model_name:
            model = AutoModel.from_pretrained(model_name)
            self.model = model.to(device=self.device)
        else:
            self.load_weights(model_name)

    def generate(
        self,
        prompts: dict[str, Any],
        **kwargs,
    ) -> list[int]:
        input_len = len(prompts["input_ids"])
        return self.model.generate(
            input_ids=torch.tensor(prompts["input_ids"], dtype=torch.int,
                                    device=self.device).unsqueeze(0),
            cross_attention_states=prompts.pop("cross_attention_states", None),
            **kwargs,
        )[0][input_len:]

    def load_weights(self, model: str) -> None:
        logger.info("Loading weights for language model...")
        comb_model = get_model(model)
        del comb_model.chunk_model
        self.model = comb_model.to(device=self.device)
        self.model.eval()
        logger.info("Finished loading.")