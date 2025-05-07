import torch
import deepspeed
from transformers import LLamaConfig, LLamaForCausalLM, AdamW
from torch.utils.data import DataLoader, dataloader_to_step
import numpy as np
import json
import os

from ..models.CombLlama import CombLlamaConfig, CombLlamaForCausalLM
from ..data.Xsum import XSumDataset

# configs
num_train_epochs = 3
model_name = "../models/Llama-3.1-8B-Instruct"
# initialize the model
model = CombLlamaForCausalLM.from_pretrained(model_name, config=CombLlamaConfig())
model = model.to('cuda')
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config_file="ds_config.json",
    optimizer=AdamW(model.parameters(), lr=1e-4)
)
# initialize the dataset and dataloader
dataset = XSumDataset()
data_loader = DataLoader(dataset, batch_size=1, num_workers=os.cpu_count()-1,
                        collate_fn=dataset.collate_fn, shuffle=True)
loss_history = []
epoch = 0
# load checkpoint if exists
if os.path.exists("checkpoints"):
    _, custom_checkpoint = model_engine.load_checkpoint("checkpoints")
    step = custom_checkpoint["step"]
    epoch = custom_checkpoint["epoch"]
    dataloader_to_step(data_loader, step + 1)
    model_engine.load_state_dict(custom_checkpoint["model"])
    optimizer.load_state_dict(custom_checkpoint["optimizer"])

while epoch < num_train_epochs:
    epoch += 1
    model_engine.train()
    for step, batch in enumerate(data_loader):
        input_ids = batch["input_ids"].to(model_engine.device)
        attention_mask = batch["attention_mask"].to(model_engine.device)
        labels = batch["labels"].to(model_engine.device)
        outputs = model_engine(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        loss_history.append(loss.item())
        model_engine.backward(loss)
        model_engine.step()
        if step % 100 == 0:
            with open("training_loss.csv", "a") as f:
                f.write(f"{epoch},{step},{loss.item()}\n")
            
        if step % 500 == 0:
            trainable_params = {k: v for k, v in model.state_dict().items() if v.requires_grad}
            custom_checkpoint = {
                "model": trainable_params,
                "optimizer": optimizer.state_dict(),
                "step": step,
                "epoch": epoch
            }
            model_engine.save_checkpoint(f"./checkpoints/", custom_checkpoint)

# Save the final model
model_engine.save_checkpoint("./final_checkpoint")
model.save_pretrained("../models/final_model")