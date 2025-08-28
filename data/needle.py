from datasets import Dataset
from functools import lru_cache
import glob
import numpy as np
import os

from data.LongBench import BenchmarkDataset

class NeedleDataset(BenchmarkDataset):
    name = "Needle-in-a-haystack"
    instruction = ("Answer the question based on the given passages. "
        "Only give me the answer and do not output any other words.\n\nQuestion: ")
    def _init_data(self, split):
        """
        Construct the dataset from the haystack and needle.

        :param needle: The needle to be found in the haystack.
        :param haystack_dir: The directory of text files to use as background context (or a haystack) in which the needle is to be found. Default is Paul Graham Essays.
        :param retrieval_question: The question which with to prompt the model to do the retrieval.
        :param final_context_length_buffer: The amount of cushion you'd like to leave off the input context to allow for the output context. Default 100 tokens
        :param context_lengths_min: The minimum length of the context. Default is 1000.
        :param context_lengths_max: The maximum length of the context. Default is 10000.
        :param context_lengths_num_intervals: The number of intervals for the context length. Default is 10.
        :param document_depth_percent_min: The minimum depth percent of the document. Default is 0.
        :param document_depth_percent_max: The maximum depth percent of the document. Default is 100.
        :param document_depth_percent_intervals: The number of intervals for the document depth percent. Default is 10.
        """
        retrieval_question = 'What is the best thing to do in San Francisco?'
        answer = 'eat a sandwich and sit in Dolores Park on a sunny day.'
        self.haystack_dir = "PaulGrahamEssays"
        self.needle = 'The best thing to do in San Francisco is to eat a sandwich and sit in Dolores Park on a sunny day.'
        self.final_context_length_buffer = 100
        self.context_lengths_min = 1000
        self.context_lengths_max = 10000
        self.context_lengths_num_intervals = 10
        self.document_depth_percent_min = 0
        self.document_depth_percent_max = 100
        self.document_depth_percent_intervals = 10
        self.context_lengths = np.round(np.linspace(
                self.context_lengths_min,
                self.context_lengths_max,
                num=self.context_lengths_num_intervals,
                endpoint=True
            )).astype(int)
        self.document_depth_percents = np.linspace(
                self.document_depth_percent_min,
                self.document_depth_percent_max,
                num=self.document_depth_percent_intervals,
                endpoint=True)
        
        data = Dataset.from_dict({
            'context': [self.generate_context(context_length, depth_percent)
                        for context_length in self.context_lengths
                        for depth_percent in self.document_depth_percents],
            'input': [retrieval_question for _ in self.context_lengths
                        for _ in self.document_depth_percents],
            'answers': [[answer] for _ in self.context_lengths
                        for _ in self.document_depth_percents],
            'context_length': [context_length for context_length in self.context_lengths
                        for _ in self.document_depth_percents],
            'depth_percent': [depth_percent for _ in self.context_lengths
                        for depth_percent in self.document_depth_percents]
        })
        self.data = data.map(self._prepare_input)
    
    def generate_context(self, context_length, depth_percent):
        # Load up tiktoken so we navigate tokens more easily

        # Get your haystack dir files loaded into a string
        context = self.read_context_files()

        # Truncate the haystack dir essays to the context length you desire
        context = self.encode_and_trim(context, context_length)

        # Insert your random statement according to your depth percent
        context = self.insert_needle(context, depth_percent, context_length)

        return context
    
    def insert_needle(self, context, depth_percent, context_length):
        # Remove <BOS> token
        tokens_needle = self.tokenizer.encode(self.needle)[1:]
        tokens_context = self.tokenizer.encode(context)[1:]

        # Reducing the context length by 150 buffer. This is to account for system message, the user question, and response.
        context_length -= self.final_context_length_buffer

        # If your context + needle are longer than the context length (which it will be), then reduce tokens from the context by the needle length
        if len(tokens_context) + len(tokens_needle) > context_length:
            tokens_context = tokens_context[:context_length - len(tokens_needle)]

        if depth_percent == 100:
            # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
            tokens_new_context = tokens_context + tokens_needle
        else:
            # Go get the position (in terms of tokens) to insert your needle
            insertion_point = int(len(tokens_context) * (depth_percent / 100))

            # tokens_new_context represents the tokens before the needle
            tokens_new_context = tokens_context[:insertion_point]

            # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
            period_tokens = self.tokenizer.encode('.')
            
            # Then we iteration backwards until we find the first period
            while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                insertion_point -= 1
                tokens_new_context = tokens_context[:insertion_point]

            # Once we get there, then add in your needle, and stick the rest of your context in on the other end.
            # Now we have a needle in a haystack
            tokens_new_context += tokens_needle + tokens_context[insertion_point:]

        # Convert back to a string and return it
        new_context = self.tokenizer.decode(tokens_new_context)
        return new_context

    def get_context_length_in_tokens(self, context):
        return len(self.tokenizer.encode(context))
    
    # Use this because the max context length (32K) is far less than the total length of all the essays 
    # Each time we call this function will get the same results
    @lru_cache() 
    def read_context_files(self):
        context = ''
        max_context_length = max(self.context_lengths)
        base_dir = os.path.abspath(os.path.dirname(__file__))  # Package directory

        for file in glob.glob(os.path.join(base_dir, self.haystack_dir, "*.txt")):
            with open(file, 'r') as f:
                context += f.read()
            if self.get_context_length_in_tokens(context) > max_context_length:
                break

        assert self.get_context_length_in_tokens(context) > max_context_length, f'Context length of all essays should be greater than {max_context_length}'
        return context

    def encode_and_trim(self, context, context_length):
        tokens = self.tokenizer.encode(context)[1:]  # Remove the first token (<BOS> token)
        if len(tokens) > context_length:
            context = self.tokenizer.decode(tokens[:context_length])
        
        return context

if __name__ == '__main__':
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    dataset = NeedleDataset(model_name, split="test")
    print(dataset[0])