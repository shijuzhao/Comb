import logging
from tqdm import tqdm
from typing import Any, Iterable, Union

from comb.integration import InferenceEngine
from comb.storage.pic_manager import PICManager
from comb.storage.pic_utils import PICSpec

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
        if 'Comb' not in model:
            raise ValueError("The model is not a COMB model. Please use the basic LLM engine directly.")

        if pic_memory_utilization + pbc_memory_utilization >= 1.0:
            raise ValueError(
                "GPU memory utilization must be less than 1.0. Got "
                f"{pic_memory_utilization + pbc_memory_utilization}.")

        pic_spec = PICSpec.from_pretrained(model)
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

    def generate(
        self,
        prompts: Union[Iterable[dict], dict[str, Any]],
        need_store: bool = True,
        **kwargs,
    ) -> list[list[int]]:
        if isinstance(prompts, dict):
            return [self.generate_for_single_request(prompts, **kwargs)]

        return [self.generate_for_single_request(prompt, need_store, **kwargs)
                    for prompt in tqdm(prompts, desc="Generating with COMB")]
    
    def generate_for_single_request(
        self,
        prompts: dict[str, Any],
        need_store: bool = True,
        **kwargs,
    ) -> list[int]:
        use_pic = False
        if "cross_attention_states" in prompts and \
                prompts["cross_attention_states"] is not None:
            # Look for the LLM engine.
            rank = prompts["cross_attention_states"].device
        elif "chunk_ids" in prompts and prompts["chunk_ids"] is not None:
            if len(prompts["chunk_ids"]) > self.pic_manager.max_context_len:
                logger.warning("The context is too long to serve.")
                return None

            # Fetch PIC.
            rank, prompts["cross_attention_states"], chunk_hash = \
                    self.pic_manager.fetch(prompts["chunk_ids"], need_store)
            use_pic = need_store
        else:
            # Round-robin allocation.
            rank = (self.next_rank + 1) % self.num_instances
            self.next_rank = rank

        outputs = self.engines[rank].generate(prompts, **kwargs)

        if use_pic:
            # Decrease reference count.
            self.pic_manager.expire(chunk_hash)

        return outputs