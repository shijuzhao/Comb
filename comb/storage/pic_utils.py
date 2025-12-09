from dataclasses import dataclass
import numpy as np
import torch
from typing import NamedTuple
from typing_extensions import Self
import xxhash

from comb.config import get_config

@dataclass
class CachePosition:
    """The position of PIC in the memory.
    
    For efficiency, we only record the left and right points of the position.
    Each slot is allocated for 1 token's KV cache.
    """
    # Left point
    left: int
    # Right point
    right: int
    def __len__(self):
        return self.right - self.left

    def __lt__(self, other):
        return self.left < other.left

    def split(self, length: int) -> tuple["CachePosition", "CachePosition"]:
        assert length < self.right - self.left
        middle = self.left + length
        return CachePosition(self.left, middle), CachePosition(middle, self.right)

    def to_range(self, rank: int) -> torch.Tensor:
        return torch.arange(self.left, self.right, device=torch.device(f"cuda:{rank}"))

class ChunkHash(NamedTuple):
    # Hash value of the chunk
    hash_value: int
    # Token IDs
    token_ids: tuple[int, ...]

@dataclass
class PICInfo:
    """Information about a PIC chunk."""
    # The device rank where the chunk is stored
    rank: int
    # Reference count
    ref_cnt: int
    # The number of requests to pin the memory
    pin_cnt: int
    # Positions
    cache_positions: list[CachePosition]
    # Hash value of the chunk
    chunk_hash: ChunkHash

    def __repr__(self) -> str:
        return (f"PICInfo(rank={self.rank}, "
                f"ref_cnt={self.ref_cnt}, "
                f"cache_positions={self.cache_positions}, "
                f"chunk_hash={self.chunk_hash}, "
                f"pin_memory={self.pin_memory})")

@dataclass
class PICSpec:
    """A class for specifying the position-independent cache format."""
    # Name or path of a model
    model_name: str
    # Number of cross-attention layers
    num_layers: int
    # Number of key/value heads
    num_key_value_heads: int
    # Head dimension
    head_dim: int
    # dtype
    dtype: torch.dtype
    @classmethod
    def from_pretrained(cls, model: str) -> Self:
        config = get_config(model)
        return cls(
            model_name=model,
            num_layers=len(config.cross_attention_layers),
            num_key_value_heads=config.text_config.num_key_value_heads,
            head_dim=config.text_config.head_dim,
            dtype=config.torch_dtype,
        )

    def get_shape(self, num_tokens: int) -> torch.Size:
        """The shape of PIC for one layer."""
        return torch.Size((2, 1, self.num_key_value_heads, num_tokens, self.head_dim))

    @property
    def size(self) -> int:
        """The size of single token's PIC."""
        return 2 * self.num_layers * self.num_key_value_heads * self.head_dim \
                * torch.tensor([], dtype=self.dtype).element_size()

def hash_tokens(tokens: list[int]) -> ChunkHash:
    """Compute a simple hash for a list of tokens."""
    hash_value = xxhash.xxh32(np.array(tokens, dtype=np.int32).tobytes()).intdigest()
    return ChunkHash(hash_value=hash_value, token_ids=tuple(tokens))

def merge_position(cp: list[CachePosition]) -> list[CachePosition]:
    """Merge the cache positions if they are continuous."""
    cp.sort()
    merged_cp = [cp.pop()]
    while cp:
        cp_new = cp.pop()
        if merged_cp[-1].right == cp_new.left:
            # If the positions are adjacent, merge them.
            merged_cp[-1].right = cp_new.right
        else:
            merged_cp.append(cp_new)

    return merged_cp
    