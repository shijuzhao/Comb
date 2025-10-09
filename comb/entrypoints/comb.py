from collections.abc import Sequence
from vllm import LLM
from vllm.inputs import PromptType
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from typing import Optional, Union

from comb.inputs import ChunkPrompt
from comb.pic_manager import PICManager

logger = init_logger(__name__)

class COMB:
    def __init__(
        self,
        pic_memory_utilization: float = 0.6,
        vllm_memory_utilization: float = 0.3,
        pic_separated: bool = False,
        **vllm_kwargs,
    ) -> None:
        if pic_memory_utilization + vllm_memory_utilization <= 1.0:
            raise ValueError(
                "GPU memory utilization must be less than 1.0. Got "
                f"{pic_memory_utilization + vllm_memory_utilization}.")

        self.pic_manager = PICManager()
        
        vllm_kwargs['gpu_memory_utilization'] = vllm_memory_utilization
        self.llm = LLM(**vllm_kwargs)

    def generate(
        self,
        prompts: Union[Union[PromptType, Sequence[PromptType]],
                       Optional[Union[str, list[str]]]] = None,
        **vllm_kwargs,
    ) -> list[RequestOutput]:
        if isinstance(prompts, ChunkPrompt) and \
            prompts.cross_attention_states is None and prompts.chunk_ids is not None:
            prompts.cross_attention_states = self.pic_manager.fetch(prompts.chunk_ids)

        return self.llm.generate(prompts, **vllm_kwargs)