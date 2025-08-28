"""This file preprocesses the LongBench dataset."""

from datasets import load_dataset

from data.base import DatasetBase, CPU_NUM

class BenchmarkDataset(DatasetBase):
    def _init_data(self, split):
        self.data = load_dataset("THUDM/LongBench", self.name, split=split,
                                trust_remote_code=True)
        self.data = self.data.map(self._prepare_input, num_proc=CPU_NUM)

    def _prepare_input(self, example):
        # Normally, the question and context are concatenated as input.
        normal_input = self.tokenizer.apply_chat_template(
            [{
                "role": "user",
                "content": self.instruction+example["input"]+example["context"]
            }]
        )
        # Split question and context
        question = self.tokenizer.apply_chat_template(
            [{
                "role": "user",
                "content": self.instruction + example["input"]
            }]
        )
        context = self.tokenizer(example["context"])
        answer = self.tokenizer.apply_chat_template(
            [{
                "role": "assistant",
                "content": example["answers"][0],
            }]
        )
        return {
            "normal_input": normal_input,
            "input_ids": question,
            "chunk_ids": context["input_ids"],
            "cross_attention_mask": context["attention_mask"],
            "labels": answer,
            "token_count": len(context["input_ids"])
        }
    
class HotpotQADataset(BenchmarkDataset):
    name = "hotpotqa"
    instruction = ("Answer the question based on the given passages. "
        "Only give me the answer and do not output any other words.\n\nQuestion: ")

class MultiNewsDataset(BenchmarkDataset):
    name = "multi_news"
    instruction = "You are given several news passages. Write a one-page summary of all news."

class SAMSumDataset(BenchmarkDataset):
    name = "samsum"
    instruction = "Summarize the following dialogue just like the preceding examples."

class MuSiQueDataset(BenchmarkDataset):
    name = "musique"
    instruction = ("Answer the question based on the given passages. "
        "Only give me the answer and do not output any other words.\n\nQuestion: ")
    
class WikiMQADataset(BenchmarkDataset):
    name = "2wikimqa"
    instruction = ("Answer the question based on the given passages. "
        "Only give me the answer and do not output any other words.\n\nQuestion: ")
        