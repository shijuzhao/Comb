# Comb
Comb is a plug-and-play caching system for long-context LLM serving.

# Getting Started
Run the following commands to prepare the environment. We recommend appending two `export` commands to the end of `~/.bashrc`.
```bash
export PYTHONPATH=~/Comb:$PYTHONPATH
export TOKENIZERS_PARALLELISM=true
pip install -r requirements.txt
```

Install vllm.
```bash
cd vllm_v0.10.0
VLLM_USE_PRECOMPILED=1 uv pip install --editable .
```

Download parameters (the parameters of `CombLlama-10B-Instruct` and `CombDeepSeek-V2-Lite` will be soon available on huggingface). You can also train a Comb model by yourself through following our [instructions](training/README.md).