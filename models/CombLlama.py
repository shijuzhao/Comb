"""
In this file, we implement the CombLlama model, 
which is a combination of a text backbone and a chunk model.
"""

import torch
from torch import nn
from transformers import LlamaConfig
from transformers.cache_utils import Cache, DynamicCache
from transformers.configuration_utils import PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaMLP, \
            LlamaRotaryEmbedding, LlamaDecoderLayer, eager_attention_forward, rotate_half
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, \
            can_return_tuple, is_torch_flex_attn_available, logging
from typing import Callable, List, Optional, Tuple, Union

if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask
    from transformers.integrations.flex_attention import make_flex_block_causal_mask

logger = logging.get_logger(__name__)

class CombLlamaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`CombLlamaForConditionalGeneration`]. It is used to instantiate an
    CombLlama model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `LlamaConfig`):
            The config object or dictionary of the text backbone.
        chunk_token_index (`int`, *optional*, defaults to 128255):
            The chunk token index to encode the cached context.
        num_hidden_layers (`int`, *optional*, defaults to 40):
            Number of hidden layers in the Transformer encoder.
        cross_attention_layers (`List[int]`, *optional*):
            Indices of the position where we insert cross attention layers. If not specified, will default to [3, 7, 11, 15, 19, 23, 27, 31].

    Example:

    ```python
    >>> from transformers import CombLlamaForConditionalGeneration, LlamaConfig, CombLlamaTextConfig

    >>> # Initializing a Llama config
    >>> text_config = LlamaConfig()

    >>> # Initializing a CombLlama style configuration
    >>> configuration = CombLlamaConfig(text_config)

    >>> # Initializing a model from the CombLlama style configuration
    >>> model = CombLlamaForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "combllama"
    attribute_map = {
        "chunk_token_id": "chunk_token_index",
    }
    sub_configs = {"text_config": LlamaConfig}

    def __init__(
        self,
        text_config: Optional[LlamaConfig] = None,
        chunk_token_index: int = 128255,
        num_hidden_layers: int = 40,
        cross_attention_layers: Optional[List[int]] = None,
        pad_token_id: Optional[int] = 128004,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        if cross_attention_layers is None:
            cross_attention_layers = [3, 7, 11, 15, 19, 23, 27, 31]

        self.chunk_token_index = chunk_token_index
        self.num_hidden_layers = num_hidden_layers
        self.cross_attention_layers = cross_attention_layers
        if text_config is None:
            self.text_config = LlamaConfig()
            logger.info("text_config is None, using default llama config")
        elif isinstance(text_config, dict):
            self.text_config = LlamaConfig(**text_config)
        elif isinstance(text_config, LlamaConfig):
            self.text_config = text_config

        super().__init__(pad_token_id=pad_token_id,
                        tie_word_embeddings=tie_word_embeddings, **kwargs)
    
# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=0, repeats=n_rep). The hidden states go from (
    num_key_value_heads, seqlen, head_dim) to (num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class CrossAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.num_heads = self.config.num_attention_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.dropout = 0.0
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // self.num_heads
        # we should modify the layer index to avoid conflict of KV cache with the text model
        self.layer_idx = layer_idx + config.num_hidden_layers
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.q_norm = LlamaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = LlamaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x SeqLen x HiddenSize"""
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        query_states = self.q_norm(query_states)

        cos, sin = position_embeddings
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        query_states = (query_states * cos) + (rotate_half(query_states) * sin)
        if cross_attention_states is not None:
            key_states, value_states = cross_attention_states
            key_states = self.k_norm(key_states)
            if past_key_value is not None and past_key_value.get_seq_length(self.layer_idx) == 0:
                # if we have a new chunk, we update the cross key states and use it!
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                )
        elif cache_position[0] != 0:
            key_states, value_states = (
                past_key_value.layers[self.layer_idx].keys,
                past_key_value.layers[self.layer_idx].values,
            )
        else:
            raise ValueError(
                "Cross attention layer can't find neither `cross_attn_states` nor cached values for key/values!"
            )

        attention_interface: Callable = eager_attention_forward

        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and output_attentions:
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights

class CombLlamaCrossAttentionDecoderLayer(torch.nn.Module):
    """Cross-attention transformer block with tanh-gated attention and feedforward."""

    def __init__(self, config: LlamaConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.cross_attn = CrossAttention(config, layer_idx=layer_idx)

        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.cross_attn_attn_gate = torch.nn.Parameter(torch.zeros(1))

        self.mlp = LlamaMLP(config)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.cross_attn_mlp_gate = torch.nn.Parameter(torch.zeros(1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: torch.Tensor,
        cross_attention_mask: torch.Tensor,
        # full_text_row_masked_out_mask: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, attn_weights = self.cross_attn(
            hidden_states=hidden_states,
            attention_mask=cross_attention_mask,
            cross_attention_states=cross_attention_states,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + self.cross_attn_attn_gate.tanh() * hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # if full_text_row_masked_out_mask is not None:
        #     hidden_states = full_text_row_masked_out_mask[:, 0] * hidden_states  # type: ignore
        hidden_states = residual + self.cross_attn_mlp_gate.tanh() * hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs

@auto_docstring
class CombLlamaPreTrainedModel(PreTrainedModel):
    config_class = CombLlamaConfig
    base_model_prefix = ""
    supports_gradient_checkpointing = True
    _no_split_modules = [
        "CombLlamaChunkModel",
        "CombLlamaCrossAttentionDecoderLayer",
        "LlamaDecoderLayer",
    ]
    _can_compile_fullgraph = False  # static cache cannot have different shapes for each layer
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = getattr(self.config, "initializer_range", self.config.get_text_config().initializer_range)

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
        elif isinstance(module, LlamaRMSNorm):
            module.weight.data.fill_(1.0)
        elif isinstance(module, CombLlamaCrossAttentionDecoderLayer):
            module.cross_attn_attn_gate.data.zero_()
            module.cross_attn_mlp_gate.data.zero_()

    # Copied from transformers.models.gptj.modeling_gptj.GPTJModel._update_causal_mask
    def _update_causal_mask(
        self,
        attention_mask: Union[torch.Tensor, "BlockMask"],
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None
        if self.config._attn_implementation == "flex_attention":
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = make_flex_block_causal_mask(attention_mask)
            return attention_mask

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_compilable_cache = past_key_values.is_compileable if past_key_values is not None else False

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_compilable_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype = input_tensor.dtype
        sequence_length = input_tensor.shape[1]
        if using_compilable_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu", "npu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    # Copied from transformers.models.gptj.modeling_gptj.GPTJModel._prepare_4d_causal_attention_mask_with_cache_position
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=cache_position.device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask

class CombLlamaChunkModel(CombLlamaPreTrainedModel):
    config_class = CombLlamaConfig
    base_model_prefix = "chunk_model"

    def __init__(self, config: CombLlamaConfig):
        text_config = config.get_text_config()
        super().__init__(text_config)
        self.hidden_size = text_config.hidden_size
        self.num_cross_layers = len(config.cross_attention_layers)
        self.head_dim = self.hidden_size // text_config.num_attention_heads
        self.num_key_value_heads = text_config.num_key_value_heads
        self.num_key_value_groups = text_config.num_attention_heads // self.num_key_value_heads
        self.embed_tokens = nn.Embedding(text_config.vocab_size, self.hidden_size, text_config.pad_token_id)
        self.rotary_emb = LlamaRotaryEmbedding(config=text_config)
        self.k_proj = nn.ModuleList([nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
                            for _ in range(self.num_cross_layers)])
        self.v_proj = nn.ModuleList([nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
                            for _ in range(self.num_cross_layers)])
        self.post_init()

    def forward(
        self,
        chunk_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        bsz, l = chunk_ids.shape
        chunk_embeds = self.embed_tokens(chunk_ids)
        position_ids = torch.arange(l, device=chunk_ids.device).unsqueeze(0)
        cos, sin = self.rotary_emb(chunk_embeds, position_ids)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        cross_attention_states = []
        for i in range(self.num_cross_layers):
            key_states = self.k_proj[i](chunk_embeds)
            value_states = self.v_proj[i](chunk_embeds)
            key_states = key_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            # RoPE
            key_states = (key_states * cos) + (rotate_half(key_states) * sin)
            cross_attention_states.append((key_states, value_states))

        return cross_attention_states

class CombLlamaTextModel(CombLlamaPreTrainedModel):
    config_class = CombLlamaConfig
    base_model_prefix = "language_model.model"
    def __init__(self, config: CombLlamaConfig):
        text_config = config.get_text_config()
        super().__init__(text_config)
        self.padding_idx = text_config.pad_token_id
        self.vocab_size = text_config.vocab_size
        self.embed_tokens = nn.Embedding(text_config.vocab_size, text_config.hidden_size, self.padding_idx)
        self.cross_attention_layers = config.cross_attention_layers

        layers = [LlamaDecoderLayer(text_config, layer_idx)
                for layer_idx in range(config.num_hidden_layers - len(self.cross_attention_layers))]
        cross_layers = [CombLlamaCrossAttentionDecoderLayer(text_config, layer_idx)
                for layer_idx in self.cross_attention_layers]

        self.layers = nn.ModuleList(layers)
        self.cross_layers = nn.ModuleList(cross_layers)
        self.norm = LlamaRMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=text_config)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cross_attention_states: Optional[List[torch.FloatTensor]] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        # full_text_row_masked_out_mask: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        The text backbone of the CombLlama model.
        
        Args:
            cross_attention_states (`torch.FloatTensor`, *optional*):
                Output of the chunk model, used for cross-attention. This tensor contains the processed chunk tokens that
                the language model will attend to.
            cross_attention_mask (`torch.Tensor` of shape `(batch_size, max_num_chunk_tokens)`, *optional*):
                Cross-attention mask to control the interaction between text tokens and chunk tokens.
                This 2D tensor defines which chunk tokens each text token should attend to.

                For each chunk token (in max_num_chunk_tokens):
                - 1 indicates the text tokens **should attend** to the corresponding chunk token
                - 0 indicates the text tokens **should not attend** to the corresponding chunk token

                TODO: Add support for 3D cross-attention masks, which would enable the chunk to be inserted at any position.

        Returns:
            `BaseModelOutputWithPast`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer
        >>> from models.CombLlama import CombLlamaTextModel

        >>> checkpoint = "models/final_model/CombLlama-9B-Instruct"
        >>> model = CombLlamaTextModel.from_pretrained(checkpoint)
        >>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        >>> text = "If I had to write a haiku, it would be:"
        >>> inputs = tokenizer(text=text, return_tensors="pt")

        >>> output = model(**inputs)

        >>> print(output.last_hidden_state.shape)
        torch.Size([1, 13, 4096])
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        next_cache = None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Here we check whether we need to compute cross attention states.
            # Let's check if the layer is cross attention layer and if we have cross attention states
            # or cached cross attention states.
            is_cross_attention_layer = idx in self.cross_attention_layers
            have_cross_attention_cache = past_key_values is not None and past_key_values.get_seq_length(idx+len(self.layers)) > 0

            if is_cross_attention_layer and (cross_attention_states is not None or have_cross_attention_cache):
                cross_layer_id = self.cross_attention_layers.index(idx)
                layer_outputs = self.cross_layers[cross_layer_id](
                    hidden_states,
                    cross_attention_states=cross_attention_states[cross_layer_id] if cross_attention_states else None,
                    cross_attention_mask=cross_attention_mask,
                    # full_text_row_masked_out_mask=full_text_row_masked_out_mask,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
                hidden_states = layer_outputs[0]
            
            # Next, we compute the self attention layer.
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

            hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
        )

@auto_docstring
class CombLlamaForCausalLM(CombLlamaPreTrainedModel, GenerationMixin):
    config_class = LlamaConfig
    _supports_static_cache = True  # only the LLM without cross attn can do compile
    base_model_prefix = "language_model"

    def __init__(self, config):
        super().__init__(config.get_text_config())
        self.text_config = config.get_text_config()
        self.vocab_size = self.text_config.vocab_size
        self.model = CombLlamaTextModel._from_config(config)
        self.lm_head = nn.Linear(self.text_config.hidden_size, self.vocab_size, bias=False)

        self.post_init()

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cross_attention_states: Optional[List[torch.LongTensor]] = None,
        cross_attention_mask: Optional[torch.LongTensor] = None,
        # full_text_row_masked_out_mask: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, CausalLMOutputWithPast]:
        r"""
            cross_attention_states (`torch.FloatTensor`, *optional*):
                Output of the chunk model, used for cross-attention. This tensor contains the processed chunk tokens that
                the language model will attend to.
            cross_attention_mask (`torch.Tensor` of shape `(batch_size, max_num_chunk_tokens)`, *optional*):
                Cross-attention mask to control the interaction between text tokens and chunk tokens.
                This 2D tensor defines which chunk tokens each text token should attend to.

                For each chunk token (in max_num_chunk_tokens):
                - 1 indicates the text tokens **should attend** to the corresponding chunk token
                - 0 indicates the text tokens **should not attend** to the corresponding chunk token

                TODO: Add support for 3D cross-attention masks, which would enable the chunk to be inserted at any position.

            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:
            `CausalLMOutputWithPast`.
        
        Example:

        ```python
        >>> from transformers import AutoTokenizer
        >>> from models.CombLlama import CombLlamaForCausalLM

        >>> model_name = "meta-llama/Llama-3.1-8B-Instruct"
        >>> model = CombLlamaForCausalLM.from_pretrained(model_name)
        >>> tokenizer = AutoTokenizer.from_pretrained(model_name)

        >>> prompt = "If I had to write a haiku, it would be:"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=40, do_sample=True, temperature=0.6)
        >>> result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        >>> print(result)
        If I had to write a haiku, it would be: "Snowflakes gently fall" - simple, yet peaceful.
        I love the idea of snowflakes gently falling, each one
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            cross_attention_states=cross_attention_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cross_attention_mask=cross_attention_mask,
            # full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :]).float()

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )

@auto_docstring
class CombLlamaForConditionalGeneration(CombLlamaPreTrainedModel, GenerationMixin):
    def __init__(self, config: CombLlamaConfig, from_scratch: bool = False):
        """
            from_scratch (`bool`): whether to initialize the new parameters, used when training. 
        """
        super().__init__(config)
        self.vocab_size = config.text_config.vocab_size
        self.hidden_size = config.text_config.hidden_size
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.chunk_model = CombLlamaChunkModel._from_config(config)
        self.language_model = CombLlamaForCausalLM._from_config(config)
        self.post_init()
        if from_scratch:
            self.language_model = CombLlamaForCausalLM.from_pretrained(
                "meta-llama/Llama-3.1-8B-Instruct", config=config)
            # init weights for new modules
            self.chunk_model.embed_tokens.load_state_dict(
                self.language_model.model.embed_tokens.state_dict())
            for i, layer_idx in enumerate(config.cross_attention_layers):
                self.chunk_model.k_proj[i].load_state_dict(
                    self.language_model.model.layers[layer_idx].self_attn.k_proj.state_dict())
                self.chunk_model.v_proj[i].load_state_dict(
                    self.language_model.model.layers[layer_idx].self_attn.v_proj.state_dict())
                self.language_model.model.cross_layers[i].cross_attn.q_proj.load_state_dict(
                    self.language_model.model.layers[layer_idx].self_attn.q_proj.state_dict())
                self.language_model.model.cross_layers[i].cross_attn.o_proj.load_state_dict(
                    self.language_model.model.layers[layer_idx].self_attn.o_proj.state_dict())
                self.language_model.model.cross_layers[i].mlp.load_state_dict(
                    self.language_model.model.layers[layer_idx].mlp.state_dict())
            # freeze the original model parameters
            for param in self.language_model.parameters():
                param.requires_grad = False
            for param in self.chunk_model.embed_tokens.parameters():
                param.requires_grad = False
            # unfreeze the cross attention layers
            for param in self.language_model.model.cross_layers.parameters():
                param.requires_grad = True
    
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        chunk_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        cross_attention_states: Optional[List[torch.Tensor]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
            chunk_ids (`torch.LongTensor` of shape `(batch_size, max_num_chunk_tokens)`, *optional*):
                Input chunk token ids to be processed by the chunk model. These tokens are used to generate the
                `cross_attention_states` that the language model will attend to.

            cross_attention_states (`torch.FloatTensor`, *optional*):
                Output of the chunk model, used for cross-attention. This tensor contains the processed chunk tokens that
                the language model will attend to.

            cross_attention_mask (`torch.Tensor` of shape `(batch_size, max_num_chunk_tokens)`, *optional*):
                Cross-attention mask to control the interaction between text tokens and chunk tokens.
                This 2D tensor defines which chunk tokens each text token should attend to.

                For each chunk token (in max_num_chunk_tokens):
                - 1 indicates the text tokens **should attend** to the corresponding chunk token
                - 0 indicates the text tokens **should not attend** to the corresponding chunk token

                TODO: Add support for 3D cross-attention masks, which would enable the chunk to be inserted at any position.

            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:
            `CausalLMOutputWithPast`.
                
        Example:

        ```python
        >>> from transformers import AutoTokenizer
        >>> from models.CombLlama import CombLlamaForConditionalGeneration

        >>> checkpoint = "models/final_model/ClmbLlama-9B-Instruct"
        >>> model = CombLlamaForConditionalGeneration.from_pretrained(checkpoint)
        >>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        >>> prompt = "If I had to write a haiku for this one"

        >>> inputs = tokenizer(text=prompt, return_tensors="pt")
        >>> chunk = tokenizer("A stop sign in Chinatown.", return_tensors="pt")

        >>> # Generate
        >>> output = model.generate(input_ids=inputs.input_ids,
                            chunk_ids=chunk.input_ids, max_new_tokens=15)

        >>> prompt_len = inputs.input_ids.shape[-1]
        >>> generated_ids = output[:, prompt_len:]
        >>> generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        >>> print(generated_text)
        [', it would be:.\\nA stop sign in Chinatown.\\n']
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if chunk_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both chunk_ids and inputs_embeds at the same time, and must specify either one"
            )

        if chunk_ids is not None and cross_attention_states is not None:
            raise ValueError("`chunk_ids` and `cross_attention_states` cannot be provided simultaneously")

        if cross_attention_mask is not None:
            # invert the mask
            inverted_cross_attn_mask = (1.0 - cross_attention_mask).to(self.dtype)
            cross_attention_mask = inverted_cross_attn_mask.masked_fill(
                inverted_cross_attn_mask.to(torch.bool), torch.finfo(self.dtype).min
            )
            
            # reshape so it can be used by attn module
            cross_attention_mask = cross_attention_mask.unsqueeze(1).unsqueeze(1)
        # else:
        #     full_text_row_masked_out_mask = None

        if chunk_ids is not None:
            # get chunk tokens from chunk model
            cross_attention_states = self.chunk_model(chunk_ids, cross_attention_mask)

        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cross_attention_states=cross_attention_states,
            cross_attention_mask=cross_attention_mask,
            # full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        # Temporary fix to calculate the loss in main class, as the model's vocab size may be resized
        loss = None
        logits = outputs[0]

        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )
    
if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    model = CombLlamaForConditionalGeneration(from_scratch=True,
                config=CombLlamaConfig(LlamaConfig.from_pretrained(model_name)))
    model = model.to('cuda')