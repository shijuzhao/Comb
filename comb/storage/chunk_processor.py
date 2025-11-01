import torch
from typing import Union

class ChunkProcessor:
    def __init__(
        self,
        model: str,
        rank: int,
    ):
        self.device = torch.device(f"cuda:{rank}")
        self.load_weights(model)

    def load_weights(model: str):
        pass

    def process(tokens: Union[torch.Tensor, list[int]]) -> torch.Tensor:
        tokens = torch.tensor(tokens, device=self.device)
        return self.chunk_model(tokens)