import logging
import torch
from typing import Union
from vllm.utils import Counter

from comb.storage.chunk_processor import ChunkProcessor
from comb.storage.pic_allocator import PICAllocator
from comb.storage.pic_utils import PICSpec, ChunkHash, PICInfo, hash_tokens

logger = logging.getLogger(__name__)

class PICManager:
    def __init__(
        self,
        num_workers: int,
        pic_spec: PICSpec,
        memory_utilization: float = 0.6,
    ):
        self.num_workers = num_workers
        self.pic: dict[int, list[torch.Tensor]] = {}
        self.total_slots = {}
        self._initialize_pic(pic_spec, memory_utilization)
        self.allocators = [PICAllocator(self.total_slots[rank])
                                    for rank in range(num_workers)]
        self.chunk_processor = [ChunkProcessor(pic_spec.model_name, rank)
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
        tokens: Union[torch.Tensor, list[int]],
    ):
        chunk_hash = hash_tokens(tokens)
        assert chunk_hash in self.cached_hash_to_picinfo, \
                f"Tokens {tokens} not found in PICManager."

        pic_info = self.cached_hash_to_picinfo[chunk_hash]
        pic_info.decr_ref()

    def fetch(
        self,
        tokens: Union[torch.Tensor, list[int]],
        pin_memory: bool = False,
    ) -> tuple[int, torch.Tensor]:
        chunk_hash = hash_tokens(tokens)
        if chunk_hash in self.cached_hash_to_picinfo:
            pic_info = self.cached_hash_to_picinfo[chunk_hash]
            pic_info.incr_ref()
            if pin_memory:
                pic_info.pin_cnt += 1
            self.allocators[pic_info.rank].touch(chunk_hash)
            cross_attention_states = self._get_tensor(pic_info.rank,
                                            pic_info.cache_positions)
        else:
            # Allocate cache positions.
            rank = -1
            for _ in range(self.num_workers):
                # Round-robin allocation.
                self.next_rank = (self.next_rank + 1) % self.num_workers
                if self._can_allocate(self.next_rank, len(tokens)):
                    rank = self.next_rank
                    break

            if rank == -1:
                logger.warning("Failed to allocate PIC.")
                return -1, None

            cache_positions = self.allocators[rank].allocate(len(tokens))
            pic_info = PICInfo(
                rank=rank,
                ref_cnt=1,
                pin_cnt=1,
                cache_positions=cache_positions,
                chunk_hash=chunk_hash,
            )
            self.cached_hash_to_picinfo[chunk_hash] = pic_info
            self.allocators[rank].register(pic_info)
            cross_attention_states = self._get_tensor(rank, cache_positions)
            # Compute the cross-attention states.
            cross_attention_states = self.chunk_processor[rank].process(tokens)

        return pic_info.rank, cross_attention_states

    def reset():
        self.cached_hash_to_picinfo: dict[ChunkHash, PICInfo] = {}
        self.next_rank = -1
        for allocator in self.allocators:
            allocator.reset()

    def shutdown():
        pass

    def unpin(
        self,
        tokens: Union[torch.Tensor, list[int]],
    ):
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
        indices = [cp.to_range(rank) for cp in cache_positions]
        return [pt[indices] for pt in self.pic[rank]]

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
            pic = []
            for _ in range(pic_spec.num_layers):
                pic.append(torch.zeros(
                    pic_spec.get_shape(self.total_slots[rank]), 
                    dtype=pic_spec.dtype, 
                    device=device
                ))

            self.pic[rank] = pic