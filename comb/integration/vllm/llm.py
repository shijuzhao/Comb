# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the COMB project
import logging
from typing import Any, Optional
from vllm import LLM
# from vllm.config.compilation import CompilationConfig, CUDAGraphMode
import uuid

from comb.transfer.cuda_ipc_utils import ProducerIpc
from comb.output import RequestOutput
from comb.supported_models import COMB_MODEL_MAPPING

logger = logging.getLogger(__name__)

MASK_32_BITS = 0xFFFFFFFF

class VLLMEngine:
    def __init__(
        self,
        model_name,
        rank,
        **kwargs,
    ) -> None:
        self.rank = rank
        self.pic_sender = ProducerIpc(rank=rank)
        self.llm = LLM(
            COMB_MODEL_MAPPING[model_name],
            seed=0,
            tokenizer=model_name,
            enable_mm_embeds=True,
            enable_prefix_caching=False,
            max_model_len=4096,
            mm_processor_cache_gb=0,
            skip_mm_profiling=True,
            skip_tokenizer_init=True,
            enforce_eager=True,
            # TODO: Enable CUDA graph by custom operator.
            # compilation_config=CompilationConfig(cudagraph_mode=CUDAGraphMode.NONE),
            # cudagraph_capture_sizes=[1],
            **kwargs,
        )
        self.sampling_params = self.llm.get_default_sampling_params()

    def __del__(self):
        if hasattr(self, "llm"):
            del self.llm
        if hasattr(self, "pic_sender"):
            self.pic_sender.close()
    
    def generate(
        self,
        prompt: dict[str, Any],
        **kwargs,
    ) -> RequestOutput:
        if "sampling_params" not in kwargs:
            kwargs["sampling_params"] = self.sampling_params.clone()

        if max_tokens := kwargs.pop("max_tokens", None):
            kwargs["sampling_params"].max_tokens = max_tokens

        if temperature := kwargs.pop("temperature", None):
            kwargs["sampling_params"].temperature = temperature

        prompt["prompt_token_ids"] = prompt.get("input_ids")
        num_prompt_tokens = len(prompt["prompt_token_ids"])
        
        # Handle cross_attention_states (PIC).
        if cross_attention_states := prompt.pop("cross_attention_states", None):
            num_prompt_tokens += cross_attention_states[0][0].shape[1]
            request_id = uuid.uuid4().int & MASK_32_BITS
            prompt["multi_modal_uuids"] = {"image": str(request_id)}
            if hasattr(cross_attention_states[0][0], 'device'):
                # PIC is a tensor on GPU, use zero-copy transfer.
                # The actual transfer will happen in the model forward.
                self.pic_sender.send_pic(cross_attention_states, request_id=str(request_id))
                prompt["multi_modal_data"] = {
                    "image": {
                        "pic_request_id": request_id,
                    }
                }
            else:
                # `cross_attention_states` is not a tensor on GPU,
                # send it as multi_modal_data directly.
                prompt["multi_modal_data"] = {
                    "image": {
                        "image_embeds": cross_attention_states,
                    }
                }
        
        vllm_out = self.llm.generate(prompt, use_tqdm=False, **kwargs)[0]
        output = RequestOutput(
            token_ids=vllm_out.outputs[0].token_ids,
            finish_reason=str(vllm_out.outputs[0].finish_reason),
        )
        if m := vllm_out.metrics:
            output.num_prompt_tokens = num_prompt_tokens
            output.num_generation_tokens = m.num_generation_tokens
            output.first_token_ts = m.first_token_ts
            output.last_token_ts = m.last_token_ts
            output.first_token_latency = m.first_token_latency

        return output
    
    def set_sampling_params(
        self,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> None:
        self.sampling_params.max_tokens = max_tokens
        self.sampling_params.temperature = temperature