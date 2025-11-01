import logging
from collections.abc import Sequence
from vllm import LLM
from vllm.inputs import PromptType
from vllm.outputs import RequestOutput
from typing import Optional, Union

from comb.inputs import ChunkPrompt
from comb.pic_manager import PICManager
from comb.storage.pic_utils import PICSpec

logger = logging.getLogger(__name__)

class COMB:
    """A system for managing position-independent cache (PIC).

    This class includes a PIC manager and an LLM inference system (vLLM).
    If a reference is detected in the prompt, COMB will fetch its PIC and
    send them to vLLM.

    Args:
        pic_memory_utilization: The ratio (between 0 and 1) of GPU memory for PIC.
            Higher values will increase the possibility of PIC reuse.
        vllm_memory_utilization: The ratio (between 0 and 1) of GPU memory for vLLM.
            This part of memory includes the prefix-based KV cache of LLM. We
            disable reusing prefix cache, so only running requests will use this
            part of memory. The total GPU memory utilization (pic_memory_utilization
            + vllm_memory_utilization) must be less than 1.
        pic_separated: Whether to assign different GPUs for PIC manager and vLLM.
    """
    def __init__(
        self,
        model: str,
        num_instances: int = 1,
        pic_memory_utilization: float = 0.6,
        vllm_memory_utilization: float = 0.3,
        pic_separated: bool = False,
        **vllm_kwargs,
    ) -> None:
        if 'Comb' not in model:
            logger.warning("The model is not a COMB model. Please use vLLM for simplicity.")

        if pic_memory_utilization + vllm_memory_utilization <= 1.0:
            raise ValueError(
                "GPU memory utilization must be less than 1.0. Got "
                f"{pic_memory_utilization + vllm_memory_utilization}.")

        pic_spec = PICSpec.from_pretrained(model)
        self.pic_manager = PICManager(num_instances, pic_spec, pic_memory_utilization)
        
        # Disable prefix caching.
        kwargs['enable_prefix_caching'] = False
        vllm_kwargs['gpu_memory_utilization'] = vllm_memory_utilization
        self.llm = LLM(model, **vllm_kwargs)

    def generate(
        self,
        prompts: Union[Union[PromptType, Sequence[PromptType]],
                       Optional[Union[str, list[str]]]] = None,
        **vllm_kwargs,
    ) -> list[RequestOutput]:
        if hasattr(prompts, "chunk_ids") and \
            prompts.cross_attention_states is None and prompts.chunk_ids is not None:
            rank, prompts.cross_attention_states = self.pic_manager.fetch(prompts.chunk_ids)

        return self.llm.generate(prompts, **vllm_kwargs)