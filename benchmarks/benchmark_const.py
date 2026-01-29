import numpy as np
# We delete the prefix of chat template for scoring.
CHAT_TEMPLATE_PREFIX = {
    "meta-llama/Llama-3.1-8B-Instruct": "assistant\n\n",
    "deepseek-ai/DeepSeek-V2-Lite-Chat": "Assistant:"
}

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
# MODEL_NAME = "deepseek-ai/DeepSeek-V2-Lite-Chat"

PORT = 9000

REQ_RATES = np.arange(4, 5.5, 0.5)