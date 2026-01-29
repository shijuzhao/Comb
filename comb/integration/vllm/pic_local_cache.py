# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the COMB project
import torch

_PIC_LOCAL_CACHE = None

def get_pic_local_cache() -> list[tuple[torch.Tensor, torch.Tensor]]:
    return _PIC_LOCAL_CACHE

def set_pic_local_cache(pic_local_cache: list[tuple[torch.Tensor, torch.Tensor]]):
    global _PIC_LOCAL_CACHE
    _PIC_LOCAL_CACHE = pic_local_cache

def reset_pic_local_cache():
    global _PIC_LOCAL_CACHE
    _PIC_LOCAL_CACHE = None