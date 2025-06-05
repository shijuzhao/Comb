# Train a comb model

We use deepspeed to train the new parameters. `ds_config.json` includes the configuration of deepspeed. We launch the training with the following command.
Remember to change directory to `training` folder first (`cd training`).
```bash
deepspeed --num_gpus=4 train.py
```