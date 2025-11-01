"""This file preprocesses the Natural-Instuctions dataset."""

from datasets import load_dataset
import re

from data.base import DatasetBase, CPU_NUM

SPECIAL_INSTRUCTIONS = [
    'This task involves creating questions from a given passage that involve some kind of complex reasoning (including numerical reasoning). ',
    'You are given an elementary science question along with a few answer options. List all the words from the question that are essential and sufficient for choosing the correct answer option. Keep in mind that essential words are necessary to understand the meaning of the given question. However, non-essential words provide little information needed to select the correct answer. You can think about it like this: If the question was in another language, knowing the meaning of which words were necessary and enough for you to understand the question and answer it?. If your answer consists of multiple spans of words from the question, separate those using a comma (", "). Do not generate any word that is not present in the question. Make sure that your response is contained in the given question and the words are in the same order as the question.',
    'You are given a question-answer pair. Answer with their type. Pay attention that there may be more than one correct type, but you only have to choose one. In your responses, use of the following types: ',
    'Read the given text and if it has abusive content then indicate via "yes". Otherwise indicate via"no". ',
    'In this task, you are given commands (in terms of logical operations) to select relevant rows from the given table. Your job is to generate a natural language interpretation of the given command. ',
    "Provide the parts-of-speech tag of a word present in a sentence specified within curly braces  ( '{{ ... }}' ). The parts-of-speech tags are coarse labels that represent a category of words with similar grammatical properties. The list of part-of-speech tags i.e tagset of this corpus is - ",
    "In this task, you need to provide the parts-of-speech tag of a word present in a sentence specified within curly braces  ( '{{ ... }}' ). The parts-of-speech tags are coarse labels that represent a category of words with similar grammatical properties. The list of part-of-speech tags i.e tagset of this corpus is - ",
    'Evaluate the similarity between them and classify them into classes from 0-5 as follows:',
    "In this task, you are given a natural language interpretation of commands (consist of logical operations) to select relevant rows from the given table. Your job is to generate command (in terms of logical operations) from given natural language interpretation. Define body (contains a collection of statements that define what the this logical operator does) of each logical operator between '{}' parenthesis. Here are the definitions of logical operators that you can use while generating command: ",
    'In this task, you are given commands (in terms of logical operations) and natural interpretation of the given command to select relevant rows from the given table. Your job is to generate a label "yes" if the interpretation is appropriate for the command, otherwise generate label "no". ',
    'In this task, you are given commands (in terms of logical operations) to select relevant rows from the given table. Your job is to classify the command into one of these seven categories: (1) majority, (2) unique, (3) superlative, (4) count, (5) comparative, (6) aggregation, and (7) ordinal. ',
    'In this task, you will be given a sentence or two along with a change aspect. You should change the given text in the given aspect. Aspects are explained below:',
    'You will be given two sentences. One of them is created by paraphrasing the original one, with changes on an aspect, or using synonyms. Your task is to decide what is the difference between two sentences. Types of change are explained below:',
    'In this task, you will be given a sentence about a person. You should determine how the sentence affects how the person is perceived by most people in society. Your choices are: ',
    'In this task you will be given a list of integers. For every element in the list, if the element is positive you should multiply it by 2.  If the element is negative you should multiply it by -3. The output should be a list of integers that is the result of applying that logic to the input list.',
    'In this task, you will be given a debate topic, along with a sentence from the debate. You should classify the given sentence and choose the type of that sentence. Possible types are explained below.',
    'In this task, a passage will be given and the goal is to generate a question about temporal relations based on that passage. A temporal relation describes the relation between two things with respect to time e.g., something happens/starts/terminates/... immediately/shortly/usually/... before/after/during/... something else.'
]

class SuperNIDataset(DatasetBase):
    name = "Super-Natural-Instructions"
    def _init_data(self, split):
        self.data = load_dataset("andersonbcdefg/supernatural-instructions-2m", split=split)
        self.data = self.data.map(self._prepare_input, num_proc=CPU_NUM)

    def _prepare_input(self, example):
        # Normally, the instruction and context are concatenated as input.
        normal_input = self.tokenizer.apply_chat_template(
            [{
                "role": "user",
                "content": example["prompt"]
            }]
        )
        # Split instruction and context
        p = re.sub(r'\n+', '\n', example["prompt"])
        s = p.split('\n')
        if s[0] in SPECIAL_INSTRUCTIONS:
            instruction = '\n'.join(s[:-1])
            context = s[-1]
        elif ((s[0][-1] == "'" and s[1][0] == "'") or s[1][0] == '.' or
            s[0] == "In this task, you're given an input that contains two words, and your task is to predict the correct preposition that connects these two words. Predict the preposition that explicitly conveys both words' meaning, e.g., Word1: king "):
            instruction = '\n'.join(s[:2])
            context = '\n'.join(s[2:])
        else:
            instruction = s[0]
            context = '\n'.join(s[1:])

        task = self.tokenizer.apply_chat_template(
            [{
                "role": "user",
                "content": instruction
            }]
        )
        context = self.tokenizer(context)
        answer = self.tokenizer.apply_chat_template(
            [{
                "role": "assistant",
                "content": example["response"],
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
    dataset = SuperNIDataset(model_name, split="train")
    print(dataset[0])