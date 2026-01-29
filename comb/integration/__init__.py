# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the COMB project
import os

from comb.integration.hf import HFEngine
from comb.integration.vllm import VLLMEngine

engine_type = os.getenv('INFERENCE_ENGINE', 'vllm')
try:
    if engine_type == 'vllm':
        import vllm
        InferenceEngine = VLLMEngine
    elif engine_type == 'sglang':
        import sglang
        raise NotImplementedError('SGLang is not ready.')
    else:
        InferenceEngine = HFEngine

except:
    InferenceEngine = HFEngine