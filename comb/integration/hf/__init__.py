# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the COMB project
import logging
import torch
from transformers import AutoModel
from typing import Any

from comb.output import RequestOutput
from comb.supported_models import COMB_MODEL_MAPPING

logger = logging.getLogger(__name__)

class HFEngine:
    def __init__(
        self,
        model_name,
        rank,
        **kwargs,
    ) -> None:
        self.device = f"cuda:{rank}"
        self.do_sample = True
        self.load_weights(COMB_MODEL_MAPPING[model_name])

    def generate(
        self,
        prompt: dict[str, Any],
        **kwargs,
    ) -> RequestOutput:
        if 'do_sample' not in kwargs:
            kwargs['do_sample'] = self.do_sample

        input_len = len(prompt["input_ids"])
        token_ids = self.model.generate(
            input_ids=torch.tensor(prompt["input_ids"], dtype=torch.int,
                                    device=self.device).unsqueeze(0),
            cross_attention_states=prompt.get("cross_attention_states", None),
            max_new_tokens=kwargs.pop("max_tokens", None),
            **kwargs,
        )[0][input_len:]
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return RequestOutput(token_ids=token_ids, finish_reason="stop")

    def load_weights(self, model: str) -> None:
        logger.info("Loading weights for language model...")
        comb_model = AutoModel.from_pretrained(model, dtype=torch.bfloat16)
        del comb_model.chunk_model
        self.model = comb_model.to(device=self.device)
        self.model.eval()
        logger.info("Finished loading.")
    
    def set_sampling_params(
        self,
        **kwargs,
    ) -> None:
        self.model.generation_config.update(**kwargs)
        # NOTE: Setting temparature in `generation_config` is not functional.
        if kwargs.get('temparature', None) == 0.0:
            self.do_sample = False