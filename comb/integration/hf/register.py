# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the COMB project

# Register models to AutoModel
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_MAPPING

from comb.integration.hf.CombLlama import CombLlamaConfig, CombLlamaForConditionalGeneration
from comb.integration.hf.CombDeepSeek import CombDeepSeekConfig, CombDeepSeekForConditionalGeneration

CONFIG_MAPPING.register("combllama", CombLlamaConfig)
CONFIG_MAPPING.register("combdeepseek", CombDeepSeekConfig)
MODEL_MAPPING.register(CombLlamaConfig, CombLlamaForConditionalGeneration)
MODEL_MAPPING.register(CombDeepSeekConfig, CombDeepSeekForConditionalGeneration)
