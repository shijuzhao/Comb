# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the COMB project
import logging
import torch
from transformers import AutoModel

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
        comb_model = AutoModel.from_pretrained(model, dtype=torch.bfloat16)
        self.chunk_model = comb_model.chunk_model.to(device=self.device)
        self.chunk_model.eval()
        del comb_model
        logger.info("Finished loading.")

    def process(self, tokens: list[int]) -> list[torch.Tensor]:
        # For SDPA attention, the behavior is unexpected if `attn_mask` is not set.
        attn_mask = torch.zeros((1, 1, 1, len(tokens)), dtype=torch.bfloat16, device=self.device)
        tokens = torch.tensor([tokens], device=self.device)
        with torch.no_grad():
            return self.chunk_model(tokens, attn_mask)