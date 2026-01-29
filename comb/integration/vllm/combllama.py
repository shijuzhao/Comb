# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the COMB project
from collections.abc import Iterable
import logging
import torch
from torch import nn
import torch.nn.functional as F
from transformers import LlamaConfig, AutoTokenizer, BatchFeature
from typing import Mapping, Optional, Sequence, Union
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.distributed import (get_tensor_model_parallel_world_size,
                              get_tensor_model_parallel_rank)
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings, SupportsLoRA, SupportsMultiModal
)
from vllm.model_executor.models.llama import LlamaMLP, LlamaDecoderLayer
from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)

from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalInputs,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    MultiModalUUIDDict,
    PlaceholderRange,
)
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (
    BaseProcessingInfo,
    EncDecMultiModalProcessor,
    PromptUpdate,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder

from comb.integration.hf.CombLlama import CombLlamaConfig
from comb.integration.vllm.pic_local_cache import (
    get_pic_local_cache, set_pic_local_cache, reset_pic_local_cache
)
from comb.transfer.cuda_ipc_utils import ConsumerIpc

logger = logging.getLogger(__name__)

class CombCrossAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        rms_norm_eps: Optional[float] = 1e-5,
        bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        if self.total_num_heads >= tp_size:
            # Number of heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_heads % tp_size == 0
        else:
            # Number of heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_heads == 0
        self.num_kv_heads = max(1, num_kv_heads // tp_size)
        self.head_dim = self.embed_dim // self.total_num_heads
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: "
                f"{self.embed_dim} and `num_heads`: {num_heads}).")
        self.scaling = self.head_dim**-0.5
        self.q_proj = ColumnParallelLinear(
            input_size=embed_dim,
            output_size=embed_dim,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )
        self.o_proj = RowParallelLinear(
            input_size=embed_dim,
            output_size=embed_dim,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: Optional[torch.Tensor],
    ):
        q, _ = self.q_proj(hidden_states)
        q = q.view(-1, self.num_heads, self.head_dim)
        q = self.q_norm(q)
        q = q.view(-1, self.num_heads * self.head_dim)

        if cross_attention_states is not None:
            k, v = cross_attention_states
            k = self.k_norm(k)
        else:
            k = v = None

        if k is None or v is None:
            # No PIC provided (or not needed for this step).
            # Keep behavior safe for warmup/capture: output zeros.
            attn_output = torch.zeros_like(q)
        else:
            # NOTE: We intentionally avoid vLLM's encoder-decoder `CrossAttention`
            # kernel here. Comb's "PIC cross-attn" is not a standard vLLM enc-dec
            # model-runner path, and during CUDA graph capture it can crash with
            # "CUDA error: invalid configuration argument" due to metadata/shape
            # mismatch. SDPA keeps everything on GPU and is robust across shapes.
            # SDPA expects (B, H, S, D). We use B=1 and token-major hidden_states.
            # q: (Sq, H, D)
            qh = q.view(-1, self.num_heads, self.head_dim).transpose(0, 1)

            # k/v: (1, Sk, Hkv, D) -> expand to (1, Sk, H, D) for GQA if needed.
            if k.shape[2] != self.num_heads:
                assert self.num_heads % k.shape[2] == 0
                rep = self.num_heads // k.shape[2]
                k = k.repeat_interleave(rep, dim=2)
                v = v.repeat_interleave(rep, dim=2)

            k_bhkd = k.squeeze(0).transpose(0, 1)
            v_bhkd = v.squeeze(0).transpose(0, 1)
            out = F.scaled_dot_product_attention(
                qh, k_bhkd, v_bhkd, dropout_p=0.0, is_causal=False
            )
            attn_output = out.transpose(0, 1).reshape(
                -1, self.num_heads * self.head_dim
            )

        output, _ = self.o_proj(attn_output)

        return output

class CombLlamaCrossAttentionDecoderLayer(nn.Module):
    """Cross-attention transformer block with tanh-gated attention
    and feedforward."""

    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: int,
        quant_config: Optional[QuantizationConfig],
        prefix: str = "",
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.cross_attn = CombCrossAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads",
                                 config.num_attention_heads),
            rms_norm_eps=config.rms_norm_eps,
            quant_config=quant_config,
            prefix=f"{prefix}.cross_attn",
        )

        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.cross_attn_attn_gate = torch.nn.Parameter(torch.zeros(1))

        self.mlp = LlamaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)
        self.cross_attn_mlp_gate = torch.nn.Parameter(torch.zeros(1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: torch.Tensor,
        # cross_attention_mask: torch.Tensor,
        # full_text_row_masked_out_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.cross_attn(
            hidden_states=hidden_states,
            cross_attention_states=cross_attention_states,
            # attention_mask=cross_attention_mask,
        )
        # hidden_states = full_text_row_masked_out_mask * hidden_states
        hidden_states = residual + self.cross_attn_attn_gate.tanh(
        ) * hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # hidden_states = full_text_row_masked_out_mask * hidden_states
        hidden_states = residual + self.cross_attn_mlp_gate.tanh(
        ) * hidden_states
        return hidden_states


@support_torch_compile
class CombLlamaTextModel(nn.Module):
    config_class = LlamaConfig
    base_model_prefix = "model"

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 prefix: str = ""):
        super().__init__()

        hf_config = vllm_config.model_config.hf_config
        config = hf_config.text_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.quant_config = quant_config
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                quant_config=quant_config,
            )
        self.cross_attention_layers = vllm_config.model_config.hf_config.cross_attention_layers
        num_cross_layers = len(self.cross_attention_layers)
        # NOTE: We modify the hf_config to initialize `LlamaDecoderLayer`
        vllm_config.model_config.hf_config = config  # type: ignore
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(vllm_config=vllm_config, prefix=f"{prefix}.layers.{idx}")
                for idx in range(config.num_hidden_layers - num_cross_layers)
        ])
        vllm_config.model_config.hf_config = hf_config  # type: ignore
        self.cross_layers = nn.ModuleList([
            CombLlamaCrossAttentionDecoderLayer(
                config=config,
                layer_idx=idx,
                quant_config=quant_config,
                prefix=f"{prefix}.cross_layers.{idx}",
            ) for idx in range(num_cross_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        cross_attention_states: Optional[list[tuple[torch.Tensor, torch.Tensor]]],
        # cross_attention_mask: Optional[torch.LongTensor],
        # full_text_row_masked_out_mask: Optional[tuple[torch.Tensor,
        #                                               torch.Tensor]],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.get_input_embeddings(input_ids)

        for idx, layer in enumerate(self.layers):
            # Insert cross-attention layers at configured indices (same convention as HF impl).
            if idx in self.cross_attention_layers and cross_attention_states is not None:
                cross_layer_id = self.cross_attention_layers.index(idx)
                hidden_states = self.cross_layers[cross_layer_id](
                    hidden_states=hidden_states,
                    cross_attention_states=cross_attention_states[cross_layer_id],
                )

            hidden_states, residual = layer(positions, hidden_states, None)
            hidden_states = hidden_states + residual
        hidden_states = self.norm(hidden_states)
        return hidden_states


class CombLlamaForCausalLM(nn.Module, SupportsLoRA):
    base_model_prefix = "language_model"
    _no_split_modules = [
        "CombLlamaCrossAttentionDecoderLayer", "LlamaDecoderLayer"
    ]

    # LoRA specific attributes
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings"
    }
    embedding_padding_modules = ["lm_head"]

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config.text_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        self.quant_config = quant_config

        self.model = CombLlamaTextModel(vllm_config=vllm_config,
                          prefix=maybe_prefix(prefix, "model"))
        
        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=(
                DEFAULT_VOCAB_PADDING_SIZE
                # We need bigger padding if using lora for kernel
                # compatibility
                if not lora_config else
                lora_config.lora_vocab_padding_size),
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(
                self.model.embed_tokens)

        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size,
                                                logit_scale)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: Optional[torch.LongTensor],
        cross_attention_states: Optional[torch.LongTensor],
        # cross_attention_mask: Optional[torch.LongTensor],
        # kv_range_for_decode: Optional[list[tuple[int, int]]],
        # full_text_row_masked_out_mask: Optional[tuple[torch.Tensor,
        #                                               torch.Tensor]],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            cross_attention_states=cross_attention_states,
            # cross_attention_mask=cross_attention_mask,
            # kv_range_for_decode=kv_range_for_decode,
            # full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if (self.quant_config is not None and
                (scale_name := self.quant_config.get_cache_scale(name))):
                # Loading kv cache quantization scales
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                loaded_weight = (loaded_weight if loaded_weight.dim() == 0 else
                                 loaded_weight[0])
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            if "scale" in name:
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if 'cross_attn' not in name:
                    name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                if 'cross_attn' in name:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)

                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
    
class CombProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> CombLlamaConfig:
        return self.ctx.get_hf_config(CombLlamaConfig)
    
    def get_mm_max_tokens_per_item(self, seq_len, mm_counts):
        return {"image": 1}

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": 1}
    
    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

class CombDummyInputsBuilder(BaseDummyInputsBuilder[CombProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return "What is the best thing to do in San Francisco?"

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        config = self.info.get_hf_config()
        num_layers = len(config.cross_attention_layers)
        num_kv_heads = config.text_config.num_key_value_heads
        head_dim = config.text_config.head_dim
        # (B, S, H, D) like COMB's PIC slices.
        a = torch.empty((1, 1, num_kv_heads, head_dim), dtype=torch.bfloat16)
        dummy_tensor = [(a, a) for _ in range(num_layers)]
        return {"image": {"image_embeds": dummy_tensor}}

class CombDummyProcessor(EncDecMultiModalProcessor[CombProcessingInfo]):
    def apply(
        self,
        prompt: Union[str, list[int]],
        mm_data: MultiModalDataDict,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Optional[Mapping[str, object]] = None,
        mm_uuids: Optional[MultiModalUUIDDict] = None,
    ) -> MultiModalInputs:
        if isinstance(prompt, str):
            tokenizer = self.info.get_tokenizer()
            processed_outputs = tokenizer(prompt,
                                          add_special_tokens=False)
            prompt = processed_outputs["input_ids"]
        
        if mm_data:
            # Check if this is a zero-copy PIC transfer
            pic_request_id = mm_data["image"].get("pic_request_id", None)
            if pic_request_id:
                # Zero-copy PIC transfer: PIC will be received via P2pNcclEngine
                # Pass the request_id and flag to the model forward
                hf_inputs = {"pic_request_id": torch.tensor([pic_request_id])}
            elif image_embeds := mm_data["image"].get("image_embeds", None):
                # Traditional method: pass PIC directly (requires serialization)
                hf_inputs = {"cross_attention_states": image_embeds}
            else:
                return {"prompt_token_ids": prompt}

            mm_kwargs = MultiModalKwargsItems.from_hf_inputs(hf_inputs,
                config_by_key=self._get_mm_fields_config(
                    hf_inputs, hf_processor_mm_kwargs)
            )
            mm_inputs = MultiModalInputs(
                type="multimodal",
                prompt_token_ids=prompt,
                mm_kwargs=mm_kwargs,
                mm_hashes={"image": [""]},
                mm_placeholders={"image": [PlaceholderRange(offset=0, length=0)]},
            )
        else:
            mm_inputs = {"prompt_token_ids": prompt}

        return mm_inputs

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        if prompt:
            tokenizer = self.info.get_tokenizer()
            processed_outputs = tokenizer(prompt,
                                          add_special_tokens=False,
                                          return_tensors="pt")
            processed_outputs.update(mm_data)
            return processed_outputs
        else:
            return mm_data

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            pic_request_id=MultiModalFieldConfig.batched("image"),
            # NOTE: Only for capturing CUDA graphs, so we set it to `shared`.
            cross_attention_states=MultiModalFieldConfig.shared("image", 1),
        )

    def create_encoder_prompt(
        self,
        prompt: Union[str, list[int]],
        mm_data: MultiModalDataDict,
    ) -> Union[str, list[int]]:
        if "chunk_ids" in mm_data:
            return mm_data["chunk_ids"]
        elif "context" in mm_data:
            tokenizer = self.info.get_tokenizer()
            return tokenizer(mm_data["context"])["input_ids"][0]
        else:
            return ""

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        return []


@MULTIMODAL_REGISTRY.register_processor(CombDummyProcessor,
                                        info=CombProcessingInfo,
                                        dummy_inputs=CombDummyInputsBuilder)
class CombLlamaForConditionalGeneration(nn.Module, SupportsMultiModal):
    merge_by_field_config = True
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"]
    }
    # NOTE: The PIC is managed by COMB, so we only support raw input.
    supports_multimodal_raw_input_only = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: CombLlamaConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.num_cross_layers = len(config.cross_attention_layers)
        self.pad_token_id = \
            config.pad_token_id if config.pad_token_id is not None else -1

        self.language_model = CombLlamaForCausalLM(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )
        self.pic_receiver = ConsumerIpc(rank=get_tensor_model_parallel_rank())

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        logits = self.language_model.logits_processor(
                    self.language_model.lm_head, hidden_states)
        return logits
    
    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        return self.language_model.get_input_embeddings(input_ids)
    
    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        cross_attention_states: Optional[list[torch.Tensor]] = None,
        # cross_attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if cross_attention_states is None:
            if (pic_request_id := kwargs.get("pic_request_id", None)):
                cross_attention_states = self.pic_receiver.receive_pic(
                    num_layers=self.num_cross_layers,
                    request_id=str(pic_request_id[0].item())
                )
                set_pic_local_cache(cross_attention_states)
            else:
                # FIXME: If the new request does not have `cross_attention_states`,
                # it will reuse the previous request's `cross_attention_states`.
                # Maybe we need `cross_attention_mask`, batching, scheduling...
                cross_attention_states = get_pic_local_cache()

        outputs = self.language_model(
            input_ids=input_ids,
            positions=positions,
            cross_attention_states=cross_attention_states,
            # cross_attention_mask=cross_attention_mask,
            # full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            inputs_embeds=inputs_embeds,
        )

        return outputs

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self, skip_prefixes=["chunk_model."])
        return loader.load_weights(weights)