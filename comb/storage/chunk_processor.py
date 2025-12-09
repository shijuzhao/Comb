import logging
import torch

from comb.config import get_model

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
        comb_model = get_model(model)
        self.chunk_model = comb_model.chunk_model.to(device=self.device)
        self.chunk_model.eval()
        del comb_model
        logger.info("Finished loading.")

    def process(self, tokens: list[int]) -> list[torch.Tensor]:
        # For SDPA attention, the behavior is unexpected if `attn_mask` is not set.
        attn_mask = torch.zeros((1, 1, 1, len(tokens)), dtype=torch.bfloat16, device=self.device)
        tokens = torch.tensor([tokens], device=self.device)
        return self.chunk_model(tokens, attn_mask)