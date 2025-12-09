"""This file generates the answers to datasets.

We expect that Comb model behaves the same as its backbone model,
so the output of backbone model is used to train.
"""

from vllm import LLM

from data import DATASET_DICT
from data.base import HF_HOME

def generate_answer(model_name):
    llm = LLM(model=model_name, tensor_parallel_size=2)
    sampling_params = llm.get_default_sampling_params()
    sampling_params.max_tokens = 512
    sampling_params.temperature = 0
    max_len = llm.get_tokenizer().model_max_length
    for name, dataset in DATASET_DICT.items():
        ds = dataset(model_name, split="train").data
        if model_name not in ds.column_names:
            # Ignore lengthy input
            prompt_token_ids = [i if len(i) < max_len else [0]
                                for i in ds["normal_input"]]
            # Generate answer using the backbone model
            output = llm.generate(
                [{"prompt_token_ids": token_ids} for token_ids in prompt_token_ids],
                sampling_params=sampling_params,
            )
            answer = [o.outputs[0].token_ids for o in output]
            ds = ds.add_column(model_name, answer)
            ds.save_to_disk(HF_HOME + f'/datasets/{name}_{model_name.replace('/', '_')}')

if __name__ == '__main__':
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    generate_answer(model_name)