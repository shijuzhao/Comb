# Train a Comb model

## Prepare datasets

We expect that Comb model behaves the same as its backbone model, so the output of backbone model is used to train. Use the script `construct_data.py` to generate answers. Remember to specify the `model_name`.
```bash
cd ~/Comb
python data/construct_data.py
```

Since the output of `Deepseek-V2-Lite` is unsatisfactory, we use the anwsers of `Llama-3.1-8B-Instruct` to train `CombDeepseek-V2-Lite` (a.k.a. distillation). Use the script `distill_data.py`.
```bash
python data/distill_data.py
```

## Adjust batch size

To prevent Out of Memory (OOM) errors, the batch size of dataset should be specified. We divide the dataset into buckets based on the length of the context. This helps in efficient batching during training. So you should adjust `BUCKET_BATCH_SIZE` in `data/base.py` according to hardware constraints. For example, the default value is for training `CombLlama-10B-Instruct` with A100 80GB GPU.

## Launch

We use deepspeed to train the new parameters. `ds_config.json` includes the configuration of deepspeed. We launch the training with the following command.
Remember to change directory to `training` folder first (`cd training`).
```bash
deepspeed --num_gpus=4 train.py
```