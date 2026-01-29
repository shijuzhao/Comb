# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the COMB project
from vllm.model_executor.models.registry import ModelRegistry

from comb.integration.vllm.combllama import CombLlamaForConditionalGeneration
from comb.integration.vllm.combdeepseek import CombDeepseekForConditionalGeneration

ModelRegistry.register_model("CombLlamaForConditionalGeneration", CombLlamaForConditionalGeneration)
ModelRegistry.register_model("CombDeepSeekForConditionalGeneration", CombDeepseekForConditionalGeneration)