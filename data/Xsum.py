from torch.utils.data import Dataset, DataLoader
from dataset import load_dataset
import os
from transformers import AutoTokenizer

class XsumDataset(Dataset):
    def __init__(self, model_name, split="train", max_length=2048):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.data = load_dataset("EdinburghNLP/xsum", split=split)
        self.data = self.data.map(self.preprocess, batched=True)

    def preprocess(self, examples):
        model_inputs = self.tokenizer("You are an AI assistant. "
                "Read the provided text and produce a concise summary. "
                "Capture the main points without unnecessary details.")
        chunks = self.tokenizer(examples["document"], max_length=self.max_length,
                                        truncation=True, padding="max_length")
        labels = self.tokenizer(examples["summary"], max_length=256,
                                        truncation=True, padding="max_length")
        model_inputs["chunk_ids"] = chunks["input_ids"]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "input_ids": item["input_ids"],
            "chunk_ids": item["chunk_ids"],
            "labels": item["labels"]
        }

    def collate_fn(self, batch):
        return {
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "chunk_ids": torch.stack([item["chunk_ids"] for item in batch]),
            "labels": torch.stack([item["labels"] for item in batch])
        }

if __name__ == "__main__":
    model_name = "../models/Llama-3.1-8B-Instruct"
    train_dataset = XsumDataset(model_name)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=train_dataset.collate_fn, num_workers=os.cpu_count()-1)
    for batch in train_dataloader:
        print(batch)
        break