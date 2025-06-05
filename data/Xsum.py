"""This file preprocesses the Xsum dataset."""

from torch import tensor, cat, stack
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer

class XsumDataset(Dataset):
    def __init__(self, model_name, split="train", max_length=2048):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, add_eos_token=False)
        if self.tokenizer.pad_token is None:
            # add new pad_token
            self.tokenizer.pad_token = '<PAD>'
            self.tokenizer.pad_token_id = 128004
        self.max_length = max_length
        self.data = load_dataset("EdinburghNLP/xsum", split=split)
        self.data = self.data.map(self.preprocess, batched=True)

    def preprocess(self, examples):
        system_prompt = self.tokenizer("You are an AI assistant. "
                "Read the provided text and produce a concise summary. "
                "Capture the main points without unnecessary details."
                , truncation=False, padding=False, return_tensors="pt")
        chunks = self.tokenizer(examples["document"], max_length=self.max_length,
                                            truncation=True, padding="max_length")
        labels = self.tokenizer(examples["summary"], max_length=256,
                                            truncation=True, padding="max_length")
        input_ids = []
        label_ids = []
        input_len = len(system_prompt["input_ids"][0]) + 1
        end_of_sentence = tensor([self.tokenizer.eos_token_id])
        # concatenate input and label
        for label in labels["input_ids"]:
            label_pt = tensor(label)
            input_ids.append(cat([system_prompt["input_ids"][0],
                                    end_of_sentence, label_pt[:-1]]))
            label_pt[label_pt == self.tokenizer.pad_token_id] = -100
            label_ids.append(cat([tensor([-100] * input_len), label_pt[1:]]))

        return {
            "input_ids": stack(input_ids),
            "chunk_ids": chunks["input_ids"],
            "labels": stack(label_ids)
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, batch):
        return {
            "input_ids": stack([tensor(item["input_ids"]) for item in batch]),
            "chunk_ids": stack([tensor(item["chunk_ids"]) for item in batch]),
            "labels": stack([tensor(item["labels"]) for item in batch])
        }

if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    dataset = XsumDataset(model_name, split="test")
    print(dataset[0])