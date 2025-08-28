"""This file preprocesses the XSum dataset."""

from datasets import load_dataset

from data.base import DatasetBase, CPU_NUM

class XsumDataset(DatasetBase):
    name = "XSum"
    def _init_data(self, split):
        self.instruction = ("You are an AI assistant. "
                "Read the provided text and produce a concise summary. "
                "Capture the main points without unnecessary details.")
        self.data = load_dataset("EdinburghNLP/xsum", split=split)
        self.input_ids = self.tokenizer.apply_chat_template(
            [{
                "role": "user",
                "content": self.instruction
            }]
        )
        self.data = self.data.map(self._prepare_input, num_proc=CPU_NUM)

    def _prepare_input(self, example):
        # Normally, the instruction and document are concatenated as input.
        normal_input = self.tokenizer.apply_chat_template(
            [{
                "role": "user",
                "content": example["document"] + self.instruction
            }]
        )
        # Split instruction and document
        context = self.tokenizer(example["document"])
        summary = self.tokenizer.apply_chat_template(
            [{
                "role": "assistant",
                "content": example["summary"]
            }]
        )
        return {
            "normal_input": normal_input,
            "input_ids": self.input_ids,
            "chunk_ids": context["input_ids"],
            "cross_attention_mask": context["attention_mask"],
            "labels": summary,
            "token_count": len(context["input_ids"])
        }

if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    dataset = XsumDataset(model_name, split="validation")
    print(dataset[0])