"""This script trains a CombLlama model using DeepSpeed."""

from datasets import Dataset
import datetime
import deepspeed
from transformers import LlamaConfig
import torch
from torch.utils.data import DataLoader
import os
import json

from comb.integration.hf.CombLlama import CombLlamaConfig, CombLlamaForConditionalGeneration
from data import TRAIN_DATASETS, DATASET_DICT
from data.base import CPU_NUM, collate_fn

# Initialize the model
model_name = "meta-llama/Llama-3.1-8B-Instruct"
model = CombLlamaForConditionalGeneration(from_scratch=True,
            config=CombLlamaConfig(LlamaConfig.from_pretrained(model_name)))
world_size = int(os.getenv('WORLD_SIZE', '1'))
local_rank = int(os.getenv('LOCAL_RANK', '0'))
device = torch.device(f'cuda:{local_rank}')
# The time for preprocessing datasets may be long, so we extend the time limit.
deepspeed.init_distributed(timeout=datetime.timedelta(seconds=7200))
ds_config = json.load(open("ds_llama_config.json"))
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config=ds_config
)
model_engine.train()
global_steps = 0
# Load checkpoint if exists
if ds_config["ckpt_folder"] is not None:
    _, client_sd = model_engine.load_checkpoint(ds_config["ckpt_folder"])
    global_steps = client_sd.get("step", 0)
    name = client_sd["dataset_name"]
    TRAIN_DATASETS = TRAIN_DATASETS[TRAIN_DATASETS.index(name) + 1:]
    # Freeze the original model parameters
    for param in model_engine.language_model.parameters():
        param.requires_grad = False
    for param in model_engine.chunk_model.embed_tokens.parameters():
        param.requires_grad = False
    # Unfreeze the cross attention layers
    for param in model_engine.language_model.model.cross_layers.parameters():
        param.requires_grad = True

# Initialize the dataset and dataloader
for dataset in TRAIN_DATASETS:
    ds = DATASET_DICT[dataset](model_name, split="train")
    # Divide the dataset based on the length of context
    for bsz, file in ds.bucketing(local_rank, world_size):
        data_loader = DataLoader(
            Dataset.from_parquet(file),
            collate_fn=collate_fn,
            batch_size=bsz,
            shuffle=False, # We have already shuffled in `ds.bucketing()`
            num_workers=max(CPU_NUM // world_size, 1),
            drop_last=False
        )
        # The micro batch size of dataloaders is different,
        # so we accumulate the gradients accordingly.
        gc_step = ds_config["train_batch_size"] // model_engine.dp_world_size //  \
                ds_config["gradient_accumulation_steps"] // bsz
        for step, batch in enumerate(data_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model_engine(**batch)
            loss = outputs.loss
            model_engine.backward(loss)
            if (step + 1) % gc_step == 0:
                model_engine.step()
                global_steps += 1
                if global_steps % 100 == 0:
                    with open("training_loss.csv", "a") as f:
                        f.write(f"{dataset},{step},{loss.item()}\n")

        model_engine.step()
        # Save checkpoint and parameters
        model_engine.eval()
        with deepspeed.zero.GatheredParameters(model_engine.module.parameters(),
                                            modifier_rank=0):
            if local_rank == 0:
                output_dir = f"../model/CombLlama-11B-Instruct({loss.item()})"
                model_engine.module.save_pretrained(output_dir)

        torch.distributed.barrier(device_ids=[local_rank])
        model_engine.train()

    model_engine.save_checkpoint("checkpoints/CombLlama-11B-Instruct",
            dataset, client_state={"dataset_name": dataset, "step": global_steps})
