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
        if split == "train":
            self.data = self.data.map(self.preprocess_for_train, batched=True)
        elif split == "test":
            self.data = self.data.map(self.preprocess_for_test, batched=True)
        else:
            raise ValueError("Split must be either 'train' or 'test'.")

    def preprocess_for_train(self, examples):
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
            "cross_attention_mask": chunks["attention_mask"],
            "labels": stack(label_ids)
        }
    
    def preprocess_for_test(self, examples):
        system_prompt = self.tokenizer("You are an AI assistant. "
                "Read the provided text and produce a concise summary. "
                "Capture the main points without unnecessary details."
                , truncation=False, padding=False, return_tensors="pt")
        chunks = self.tokenizer(examples["document"], max_length=self.max_length,
                                            truncation=True, padding="max_length")
        labels = self.tokenizer(examples["summary"], max_length=256,
                                            truncation=True, padding="max_length")
        return {
            "input_ids": system_prompt["input_ids"].repeat(len(chunks["input_ids"]), 1),
            "chunk_ids": chunks["input_ids"],
            "cross_attention_mask": chunks["attention_mask"],
            "labels": labels["input_ids"]
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, batch):
        return {
            key: stack([tensor(item[key]) for item in batch])
            for key in ["input_ids", "chunk_ids", "cross_attention_mask", "labels"]
        }

if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    dataset = XsumDataset(model_name, split="test")
    print(dataset[0])