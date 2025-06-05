"""This script trains a CombLlama model on the Xsum dataset using DeepSpeed."""

import deepspeed
from transformers import LlamaConfig
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
import os
import json

from models.CombLlama import CombLlamaConfig, CombLlamaForConditionalGeneration
from data.Xsum import XsumDataset

# configs
num_train_epochs = 3
model_name = "meta-llama/Llama-3.1-8B-Instruct"
ckpt_folder = None
# initialize the model
model = CombLlamaForConditionalGeneration(from_scratch=True,
            config=CombLlamaConfig(LlamaConfig.from_pretrained(model_name)))
local_rank = int(os.getenv('LOCAL_RANK', '0'))
device = torch.device(f'cuda:{local_rank}')
model.to(device)
ds_config = json.load(open("ds_config.json"))
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config=ds_config,
    optimizer=AdamW(model.parameters(), lr=1e-4)
)
# initialize the dataset and dataloader
dataset = XsumDataset(model_name, split="train")
data_loader = DataLoader(dataset, batch_size=ds_config["train_batch_size"],
    num_workers=os.cpu_count()-1, collate_fn=dataset.collate_fn, shuffle=True)
epoch = 0
# load checkpoint if exists
if ckpt_folder is not None:
    _, custom_checkpoint = model_engine.load_checkpoint(ckpt_folder)
    step = custom_checkpoint["step"]
    epoch = custom_checkpoint["epoch"]
    # advance the dataloader to the checkpoint step
    for _ in range(step):
        try:
            next(iter(data_loader))
        except StopIteration:
            break
    model_engine.load_state_dict(custom_checkpoint["model"])
    optimizer.load_state_dict(custom_checkpoint["optimizer"])

while epoch < num_train_epochs:
    model_engine.train()
    for step, batch in enumerate(data_loader):
        input_ids = batch["input_ids"].to(model_engine.device)
        chunk_ids = batch["chunk_ids"].to(model_engine.device)
        labels = batch["labels"].to(model_engine.device)
        outputs = model_engine(
            input_ids=input_ids,
            chunk_ids=chunk_ids,
            labels=labels
        )
        loss = outputs.loss
        model_engine.backward(loss)
        model_engine.step()
        if step % 100 == 0:
            with open("training_loss.csv", "a") as f:
                f.write(f"{epoch},{step},{loss.item()}\n")
            
    epoch += 1

# Save the final model
model_engine.save_checkpoint("./final_checkpoint/CombLlama-9B-Instruct")
model.save_pretrained("../models/CombLlama-9B-Instruct")