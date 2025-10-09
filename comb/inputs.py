import torch
from typing import Optional
from vllm.inputs import TokensPrompt

class ChunkPrompt(TokensPrompt):
    chunk_ids: Optional[list[int]] = None
    """A list of token ids as reference."""

    cross_attention_states: Optional[torch.Tensor] = None
    """The cross attention states for the chunk prompt."""