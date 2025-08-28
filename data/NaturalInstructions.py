"""This file preprocesses the Natural-Instuctions dataset."""

from datasets import load_dataset

from data.base import DatasetBase, CPU_NUM

class NIDataset(DatasetBase):
    name = "Natural-Instructions"
    def _init_data(self, split):
        self.data = load_dataset("Muennighoff/natural-instructions",
                                revision="refs/convert/parquet", split=split)
        self.data = self.data.map(self._prepare_input, num_proc=CPU_NUM)

    def _prepare_input(self, example):
        # Normally, the instruction and context are concatenated as input.
        normal_input = self.tokenizer.apply_chat_template(
            [{
                "role": "user",
                "content": example["definition"] + example["inputs"]
            }]
        )
        # Split instruction and context
        task = self.tokenizer.apply_chat_template(
            [{
                "role": "user",
                "content": example["definition"]
            }]
        )
        context = self.tokenizer(example["inputs"])
        answer = self.tokenizer.apply_chat_template(
            [{
                "role": "assistant",
                "content": example["targets"],
            }]
        )
        return {
            "normal_input": normal_input,
            "input_ids": task,
            "chunk_ids": context["input_ids"],
            "cross_attention_mask": context["attention_mask"],
            "labels": answer,
            "token_count": len(context["input_ids"])
        }

if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    dataset = NIDataset(model_name, split="validation")
    print(dataset[0])
