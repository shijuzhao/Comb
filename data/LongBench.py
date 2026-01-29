"""This file preprocesses the LongBench dataset."""

from datasets import load_dataset

from data.base import DatasetBase, CPU_NUM
from data.metrics import qa_f1_score, rouge_score

class BenchmarkDataset(DatasetBase):
    def _init_data(self, split):
        self.data = load_dataset("THUDM/LongBench", self.name, split='test',
                                trust_remote_code=True)
        if hasattr(self, 'correct_answers'):
            # We find that some answers are incorrect, so we correct them here.
            self.data = self.data.map(self._correct, num_proc=CPU_NUM)

        self.data = self.data.map(self._prepare_input, num_proc=CPU_NUM)

    def _correct(self, example):
        if example["input"] in self.correct_answers:
            example["answers"] = self.correct_answers[example["input"]]
        return example

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
        prompt = {
            "normal_input": normal_input,
            "input_ids": question,
            "chunk_ids": context["input_ids"],
            "cross_attention_mask": context["attention_mask"],
            "labels": answer,
            "token_count": len(context["input_ids"])
        }
        # We modify the insturction to simulate a new request.
        if hasattr(self, "instruction_new"):
            normal_input_new = self.tokenizer.apply_chat_template(
                [{
                    "role": "user",
                    "content": self.instruction_new + \
                            example["input"] + example["context"]
                }]
            )
            question_new = self.tokenizer.apply_chat_template(
                [{
                    "role": "user",
                    "content": self.instruction_new + example["input"]
                }]
            )
            prompt["normal_input_new"] = normal_input_new
            prompt["input_ids_new"] = question_new
        
        return prompt
    
class HotpotQADataset(BenchmarkDataset):
    name = "hotpotqa"
    metric = qa_f1_score
    instruction = ("Answer the question based on the given passages. "
        "Only give me the answer and do not output any other words.\n\nQuestion: ")
    instruction_new = ("Answer the question based on the given passages. "
        "Only give me the answer and do not output any other words.\n\nThe question is: ")
    correct_answers = {
        "Which mountain is higher, Tongshanjiabu or Himalchuli?": ["Himalchuli", "Himalchuli is higher than Tongshanjiabu."],
        "Band-e-Amir Dragons is named after the lakes in which Afghan national park?": ["Band-e-Amir National Park"],
        "In what event was Harold Davis a former record holder, but now is held by Usain Bolt?": ["100 metres", "100 m"],
        "What career led Brandon James Routh to move to the city where Brian Ralston lives?": ["an acting career", "acting"],
        "What is the character of fictional character Claire Fraser in a British-American television drama series developed by Ronald D. Moore ?": ["Claire is a married World War II nurse", "A nurse."],
        "Which composer was wrote his music most recently, Michael Tippett or Luigi Cherubini?": ["Michael Tippett", "Michael Kemp Tippett"],
        "Which filmmaker was known for animation, Lev Yilmaz or Pamela B. Green?": ["Lev Yilmaz", "Lev Yilmaz was known for animation."],
    }

class MultiNewsDataset(BenchmarkDataset):
    name = "multi_news"
    metric = rouge_score
    instruction = ("You are an AI assistant. "
                "Read the provided text and produce a concise summary. "
                "Capture the main points without unnecessary details.")
    instruction_new = ("You are given several news passages. "
                        "Write a one-page summary of all news.")

class SAMSumDataset(BenchmarkDataset):
    name = "samsum"
    metric = rouge_score
    instruction = ("You are an AI assistant. "
                "Read the provided text and produce a concise summary. "
                "Capture the main points without unnecessary details.")
    instruction_new = "Summarize the following dialogue just like the preceding examples."

class MuSiQueDataset(BenchmarkDataset):
    name = "musique"
    metric = qa_f1_score
    instruction = ("Answer the question based on the given passages. "
        "Only give me the answer and do not output any other words.\n\nQuestion: ")
    instruction_new = ("Answer the question based on the given passages. "
        "Only give me the answer and do not output any other words.\n\nThe question is: ")
    correct_answers = {
        "When did the party who gained control of congress in the midterm elections in 1946 take control of the determiner of rules of the US House and US Senate?": ["January 3, 1947", "January, 1947"],
        "What pantheon is the God of the underworld in ancient Egypt a part of?": ["Egyptian pantheon", "The God of the underworld is a part of the Egyptian pantheon."],
        "What other recognition did the Oscar winner for Best Actor in 2006 receive?": ["nominated for an Academy Award for Best Supporting Actor", "Academy Award for Best Supporting Actor", "Best Supporting Actor"],
    }
    
class WikiMQADataset(BenchmarkDataset):
    name = "2wikimqa"
    metric = qa_f1_score
    instruction = ("Answer the question based on the given passages. "
        "Only give me the answer and do not output any other words.\n\nQuestion: ")
    instruction_new = ("Answer the question based on the given passages. "
        "Only give me the answer and do not output any other words.\n\nThe question is: ")
        