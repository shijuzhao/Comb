# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the COMB project
import logging
from time import perf_counter
from tqdm import tqdm
from typing import Any, Iterable, Optional, Union

from comb.integration import InferenceEngine
from comb.storage.pic_manager import PICManager
from comb.storage.pic_utils import PICSpec
from comb.supported_models import COMB_MODEL_MAPPING

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class COMB:
    """A system for managing position-independent cache (PIC).

    This class includes a PIC manager and an LLM inference system (vLLM).
    If a reference is detected in the prompt, COMB will fetch its PIC and
    send them to vLLM.

    Args:
        model: The name or path of a HuggingFace Transformers model.
        num_instances: The number of inference engines.
        pic_memory_utilization: The ratio (between 0 and 1) of GPU memory for PIC.
            Higher values will increase the possibility of PIC reuse.
        pbc_memory_utilization: The ratio (between 0 and 1) of GPU memory for
            prefix-based cache (PBC). This part of memory includes the prefix-based
            KV cache of LLM. We disable reusing prefix cache, so only running
            requests will use this part of memory. The total GPU memory utilization
            (pic_memory_utilization + pbc_memory_utilization) must be less than 1.
        pic_separated: Whether to assign different GPUs for PIC manager and LLM engine.
    """
    def __init__(
        self,
        model: str,
        num_instances: int = 1,
        pic_memory_utilization: float = 0.3,
        pbc_memory_utilization: float = 0.3,
        pic_separated: bool = False,
        **kwargs,
    ) -> None:
        if model not in COMB_MODEL_MAPPING:
            raise ValueError(f"The model {model} is not supported by COMB currently."
                    " Please check the model name or use the basic LLM engine.")

        if pic_memory_utilization + pbc_memory_utilization >= 1.0:
            raise ValueError(
                "GPU memory utilization must be less than 1.0. Got "
                f"{pic_memory_utilization + pbc_memory_utilization}.")

        pic_spec = PICSpec.from_pretrained(COMB_MODEL_MAPPING[model])
        self.pic_manager = PICManager(num_instances, pic_spec, pic_memory_utilization)
        
        self.engines = [
            InferenceEngine(
                model, rank,
                gpu_memory_utilization=pbc_memory_utilization, **kwargs)
            for rank in range(num_instances)
        ]

        # `next_rank` is used for scheduling requests without PIC.
        # We assume that the number of these requests is small,
        # and they should be directed to traditional LLM serving systems.
        self.next_rank = -1
        self.num_instances = num_instances
        self.enable_log_stats = not kwargs.get("disable_log_stats", True)

    def generate(
        self,
        prompts: Union[Iterable[dict], dict[str, Any]],
        need_store: bool = True,
        use_tqdm: bool = True,
        **kwargs,
    ) -> list[list[int]]:
        if isinstance(prompts, dict):
            return [self.generate_for_single_request(prompts, **kwargs)]

        if use_tqdm:
            prompts = tqdm(prompts, desc="Generating with COMB")

        return [self.generate_for_single_request(prompt, need_store, **kwargs)
                    for prompt in prompts]
    
    def generate_for_single_request(
        self,
        prompt: dict[str, Any],
        need_store: bool = True,
        **kwargs,
    ) -> list[int]:
        if self.enable_log_stats:
            arrival_time = perf_counter()

        use_pic = False
        if cross_attention_states := prompt.get("cross_attention_states", None):
            if isinstance(cross_attention_states, list) and len(cross_attention_states) > 0 and \
                isinstance(cross_attention_states[0], tuple) and len(cross_attention_states[0]) == 2:
                # This is a list of (key, value) tuples - PIC format
                # Check if tensors are on GPU
                if cross_attention_states[0][0].is_cuda:
                    rank = cross_attention_states[0][0].device.index
                else:
                    rank = 0  # Default rank
            else:
                logger.warning("Invalid cross_attention_states format."
                                "It will not be applied.")
                rank = 0
        elif chunk_ids := prompt.get("chunk_ids", None):
            if len(chunk_ids) > self.pic_manager.max_context_len:
                logger.warning("The context is too long to serve.")
                return []

            # Fetch PIC.
            rank, prompt["cross_attention_states"], chunk_hash = \
                    self.pic_manager.fetch(chunk_ids, need_store)
            use_pic = need_store
        else:
            # Round-robin allocation.
            rank = (self.next_rank + 1) % self.num_instances
            self.next_rank = rank

        if self.enable_log_stats:
            pic_latency = perf_counter() - arrival_time
        
        outputs = self.engines[rank].generate(prompt, **kwargs)

        if use_pic:
            # Decrease reference count.
            self.pic_manager.expire(chunk_hash)

        if self.enable_log_stats:
            outputs.first_token_latency += pic_latency
        
        return outputs
    
    def set_sampling_params(
        self,
        rank: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Set sampling parameters for inference engines.

        Args:
            kwargs: Sampling parameters, e.g., temperature, top_k, top_p, etc.
        """
        if rank is not None:
            self.engines[rank].set_sampling_params(**kwargs)
            return
        
        for engine in self.engines:
            engine.set_sampling_params(**kwargs)