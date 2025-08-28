"""This file preprocesses the SQuAD dataset."""

from datasets import load_dataset

from data.base import DatasetBase, CPU_NUM

class SquadDataset(DatasetBase):
    name = "SQuAD"
    def _init_data(self, split="train"):
        self.data = load_dataset("squad_v2", split=split)
        self.data = self.data.map(self._prepare_input, num_proc=CPU_NUM)
        
    def _prepare_input(self, example):
        # Normally, the context and question are concatenated as input.
        normal_input = self.tokenizer.apply_chat_template(
            [{
                "role": "user",
                "content": example["context"] + example["question"]
            }]
        )
        # Split question and context
        question = self.tokenizer.apply_chat_template(
            [{
                "role": "user",
                "content": example["question"]
            }]
        )
        context = self.tokenizer(example["context"])
        if len(example["answers"]["text"]) == 0:
            answer = "No information is provided in the context."
        else:
            answer = example["answers"]["text"][0]

        answer = self.tokenizer.apply_chat_template(
            [{
                "role": "assistant",
                "content": answer
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

if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    dataset = SquadDataset(model_name, split="validation")
    print(dataset[0])