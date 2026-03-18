# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the COMB project
import logging
import torch
from transformers import AutoModel
from transformers.utils import is_flash_attn_2_available

logger = logging.getLogger(__name__)

class ChunkProcessor:
    def __init__(
        self,
        model: str,
        rank: int,
    ) -> None:
        self.device = f"cuda:{rank}"
        self.load_weights(model)

    def load_weights(self, model: str) -> None:
        logger.info("Loading weights for chunk model...")
        comb_model = AutoModel.from_pretrained(model, dtype=torch.bfloat16,
                        attn_implementation="flash_attention_2"
                        if is_flash_attn_2_available() else "sdpa")
        self.chunk_model = comb_model.chunk_model.to(device=self.device)
        self.chunk_model.eval()
        del comb_model
        logger.info("Finished loading.")

    def process(self, tokens: list[int]) -> list[torch.Tensor]:
        tokens = torch.tensor([tokens], device=self.device)
        with torch.no_grad():
            return self.chunk_model(tokens)