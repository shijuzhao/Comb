It is simple to conduct the experiments. The configuration of parameters is completed in `benchmark_consts.py`. You just need to run `main.py` with different values of the argument `--experiment`.
```bash
python3 main.py
```

## Offline Experiment

`./main.py --experiment offline` for Figure 4, 6, and 7.

**Time consumed**
- `Llama-3.1-8B-Instruct`: 6 hours;
- `DeepSeek-V2-Lite-Chat`: 15 hours.

## KV Cache Size

`./main.py --experiment cache_size` for Figure 8.

## Online Experiment

`./main.py --experiment online` for Figure 9.

**Time Consumed**: 2 hours

## Benchmarking Block Attention

```bash
cd block_attention
python run_block_attention.py
```

**Time Consumed**: 2 hours