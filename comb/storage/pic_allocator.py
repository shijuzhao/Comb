from collections import deque, OrderedDict
import gc
import torch

from comb.storage.pic_utils import CachePosition, PICSpec, ChunkHash, PICInfo, \
                                    merge_position

class PICAllocator:
    def __init__(
        self,
        total_slots: int,
    ):
        self.total_slots = total_slots
        self.cached_pic = OrderedDict()
        self.reset()

    def allocate(
        self,
        num_tokens: int,
    ) -> list[CachePosition]:
        assert num_tokens <= self.num_free_slots
        self.num_free_slots -= num_tokens
        cache_positions = []
        while num_tokens:
            cp = self.free_slots.pop()
            length = len(cp)
            if num_tokens < length:
                allocated_cp, remaining_cp = cp.split(length)
                cache_positions.append(allocated_cp)
                self.free_slots.append(remaining_cp)
                break
            else:
                num_tokens -= length
                cache_positions.append(cp)

        return merge_position(cache_positions)

    def register(
        self,
        pic_info: PICInfo,
    ):
        self.cached_pic[pic_info.chunk_hash] = pic_info

    def reset(self):
        self.free_slots = deque(CachePosition(0, self.total_slots))
        self.num_free_slots = self.total_slots
        self.cached_pic.clear()

    def touch(
        self,
        chunk_hash: ChunkHash,
    ):
        self.cached_pic.move_to_end(chunk_hash)

    def preempt(
        self,
        num_tokens: int,
    ) -> tuple[bool, list[ChunkHash]]:
        preempted_pic = []
        for chunk_hash, pic_info in self.cached_pic.items():
            if pic_info.ref_cnt or pic_info.pin_cnt:
                continue

            preempted_pic.append(chunk_hash)
            for cp in pic_info.cache_positions:
                self.free_slots.appendleft(cp)

            self.num_free_slots += len(chunk_hash.token_ids)
            if self.num_free_slots >= num_tokens:
                return True, preempted_pic

        return False, preempted_pic
