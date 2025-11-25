import logging
import time
import torch
from typing import Optional, Union

from comb.storage.chunk_processor import ChunkProcessor
from comb.storage.pic_allocator import PICAllocator
from comb.storage.pic_utils import CachePosition, ChunkHash, PICInfo, PICSpec, hash_tokens

logger = logging.getLogger(__name__)

class PICManager:
    def __init__(
        self,
        num_workers: int,
        pic_spec: PICSpec,
        memory_utilization: float = 0.6,
    ) -> None:
        self.num_workers = num_workers
        self.chunk_processor = [ChunkProcessor(pic_spec.model_name, rank)
                                    for rank in range(num_workers)]
        self.pic: dict[int, list[torch.Tensor]] = {}
        self.total_slots = {}
        self._initialize_pic(pic_spec, memory_utilization)
        self.max_context_len = max(self.total_slots.values())
        self.allocators = [PICAllocator(self.total_slots[rank])
                                    for rank in range(num_workers)]
        self.reset()

    def exists(
        self,
        tokens: Union[torch.Tensor, list[int]],
    ) -> bool:
        chunk_hash = hash_tokens(tokens)
        return chunk_hash in self.cached_hash_to_picinfo

    def expire(
        self,
        chunk_hash: Optional[ChunkHash],
    ) -> None:
        assert chunk_hash in self.cached_hash_to_picinfo, \
                f"Tokens with hash value {chunk_hash} not found in PICManager."

        pic_info = self.cached_hash_to_picinfo[chunk_hash]
        pic_info.ref_cnt -= 1

    def fetch(
        self,
        tokens: Union[torch.Tensor, list[int]],
        pin_memory: bool = False,
    ) -> tuple[int, list[torch.Tensor], ChunkHash]:
        chunk_hash = hash_tokens(tokens)
        if chunk_hash in self.cached_hash_to_picinfo:
            pic_info = self.cached_hash_to_picinfo[chunk_hash]
            pic_info.ref_cnt += 1
            if pin_memory:
                pic_info.pin_cnt += 1
            self.allocators[pic_info.rank].touch(chunk_hash)
            cross_attention_states = self._get_tensor(pic_info.rank,
                                            pic_info.cache_positions)
        else:
            # Allocate cache positions.
            rank = self.schedule(len(tokens))
            cache_positions = self.allocators[rank].allocate(len(tokens))
            pic_info = PICInfo(
                rank=rank,
                ref_cnt=1,
                pin_cnt=1 if pin_memory else 0,
                cache_positions=cache_positions,
                chunk_hash=chunk_hash,
            )
            self.cached_hash_to_picinfo[chunk_hash] = pic_info
            self.allocators[rank].register(pic_info)
            # TODO(Optimize): Overlap with CPU operations.
            # Compute the cross-attention states.
            cross_attention_states = self.chunk_processor[rank].process(tokens)
            self.store(rank, cache_positions, cross_attention_states)

        return pic_info.rank, cross_attention_states, chunk_hash

    def reset() -> None:
        self.cached_hash_to_picinfo: dict[ChunkHash, PICInfo] = {}
        self.next_rank = -1
        for allocator in self.allocators:
            allocator.reset()

    def schedule(
        self,
        num_tokens: int
    ) -> int:
        for _ in range(self.num_workers):
            # Round-robin allocation.
            self.next_rank = (self.next_rank + 1) % self.num_workers
            if self._can_allocate(self.next_rank, num_tokens):
                return self.next_rank

        # NOTE: `available_workers` is not empty,
        # because lengthy requests are excluded in advance.
        available_workers = [rank for rank, total_slots in self.total_slots.items()
                                if total_slots >= num_tokens]

        # Wait until enough space is free.
        # NOTE: Maybe we need a request queue and implement complicated scheduler.
        while True:
            time.sleep(0.1)
            for rank in available_workers:
                if self._can_allocate(rank, num_tokens):
                    return rank
    
    def store(
        self,
        rank: int,
        cache_positions: list[CachePosition],
        cross_attention_states: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        indices = torch.cat([cp.to_range(rank) for cp in cache_positions], dim=0)
        for i, pic in enumerate(self.pic[rank]):
            pic[0, indices] = cross_attention_states[i][0]
            pic[1, indices] = cross_attention_states[i][1]

    def unpin(
        self,
        tokens: Union[torch.Tensor, list[int]],
    ) -> None:
        chunk_hash = hash_tokens(tokens)
        assert chunk_hash in self.cached_hash_to_picinfo, \
                f"Tokens {tokens} not found in PICManager."

        self.cached_hash_to_picinfo[chunk_hash].pin_cnt -= 1

    def _can_allocate(
        self,
        rank: int,
        num_tokens: int,
    ) -> bool:
        """Check if we can allocate PIC for num_tokens on the given rank.
            If not, try to preempt PIC whose ref_cnt is 0."""
        if self.allocators[rank].num_free_slots >= num_tokens:
            return True

        can_allocate, preempted_pic = self.allocators[rank].preempt(num_tokens)
        for chunk_hash in preempted_pic:
            del self.cached_hash_to_picinfo[chunk_hash]

        return can_allocate

    def _get_tensor(
        self,
        rank: int,
        cache_positions: list[CachePosition],
    ) -> list[torch.Tensor]:
        indices = torch.cat([cp.to_range(rank) for cp in cache_positions], dim=0)
        return [(pt[0, indices], pt[1, indices]) for pt in self.pic[rank]]

    def _initialize_pic(
        self,
        pic_spec: PICSpec,
        memory_utilization: float = 0.6,
    ) -> list[torch.Tensor]:
        for rank in range(self.num_workers):
            device = torch.device(f"cuda:{rank}")
            torch.cuda.set_device(device)
            gc.collect()
            torch.cuda.empty_cache()
            free_memory, total_memory = torch.cuda.mem_get_info()
            required_memory = total_memory * memory_utilization
            if free_memory < required_memory:
                GiB = lambda b: round(b / GiB_bytes, 2)
                raise ValueError(
                    f"Free memory on device {rank}"
                    f"({GiB(free_memory)}/{GiB(total_memory)} GiB) on startup "
                    f"is less than desired GPU memory utilization "
                    f"({memory_utilization}, {GiB(required_memory)} GiB). Decrease "
                    f"GPU memory utilization or reduce GPU memory used by other processes."
                )

            self.total_slots[rank] = int(required_memory // pic_spec.size)
            self.pic[rank] = [
                torch.empty(
                    pic_spec.get_shape(self.total_slots[rank]),
                    dtype=pic_spec.dtype,
                    device=device
                ) for _ in range(pic_spec.num_layers)
            ]