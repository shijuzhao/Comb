<p align="center">
  <picture>
    <img alt="COMB" src="assets/logo.svg" width=55%>
  </picture>
</p>

# COMB
COMB is a plug-and-play caching system for long-context LLM serving.

## Code Structure
```
COMB
├── benchmarks                   # For benchmarking
├── comb
│   ├── entrypoints
│   │   ├── api_server.py        # For online server
│   │   └── comb.py              # For offline inference
│   ├── integration
│   │   ├── hf                   # hf transformers backend
│   │   ├── vllm                 # vLLM backend
│   │   └── __init__.py
│   ├── storage
│   │   ├── chunk_processor.py   # For generating PIC
│   │   ├── pic_allocator.py     # For allocating memory
│   │   ├── pic_manager.py       # For managing PIC
│   │   └── pic_utils.py
│   ├── transfer
│   │   └── cuda_ipc_utils.py    # For inter-process communication
│   ├── __init__.py
│   ├── output.py
│   └── supported_models.py
├── data
├── examples                     # For use case
├── training                     # For training
├── environment.yml
└── requirements.txt
```

## Getting Started
Run the following commands to prepare the environment. We recommend appending two `export` commands to the end of `~/.bashrc`.
```bash
export PYTHONPATH=~/Comb:$PYTHONPATH
export TOKENIZERS_PARALLELISM=true
pip install -r requirements.txt
```

Install vllm. (Recommended for efficiency and benchmarking)
```bash
pip install vllm
```

Currently we only support `meta-llama/Llama-3.1-8B-Instruct` and `deepseek-ai/DeepSeek-V2-Lite-Chat`. If you want to use another model, you can also train a Comb model by yourself through following our [instructions](training/README.md).

## Usage

You can find examples in the folder `examples`.

- [basic.py](examples/basic.py) for offline inference.
- [online_serving.py](examples/online_serving.py) for server.

## Benchmark

See [Instructions](benchmarks/README.md).

## Demo

In this example, we simulate two requests with different prefixes. The requests contain the same question and retrieved context, enabling the KV cache to be reused through PIC.

<!-- <video width="640" height="360" controls>
  <source src="assets/demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video> -->
[video:Demo](assets/demo.mp4)