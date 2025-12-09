"""Distillation

When the capability of a model is low, we can improve it by training it with answers
from advanced LLMs. For example, we generate the answers of Llama-3.1-8B-Instruct,
tokenize them with target model's tokenizer, and finally train the target model.
"""

from transformers import AutoTokenizer
from vllm import LLM

from data import DATASET_DICT
from data.base import HF_HOME

def distill_from_llama(model_name):
    model_to_be_distilled = "meta-llama/Llama-3.1-8B-Instruct"
    llm = LLM(model=model_to_be_distilled, tensor_parallel_size=2)
    sampling_params = llm.get_default_sampling_params()
    sampling_params.max_tokens = 512
    sampling_params.temperature = 0
    max_len = llm.get_tokenizer().model_max_length
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    for name, dataset in DATASET_DICT.items():
        ds = dataset(model_name, split="train").data
        if model_name not in ds.column_names:
            data_to_be_distilled = dataset(model_to_be_distilled, split="train")
            # Ignore lengthy input
            prompt_token_ids = [i if len(i) < max_len else [0]
                for i in data_to_be_distilled["normal_input"]]
            # Generate answer using the backbone model
            output = llm.generate(
                [{"prompt_token_ids": token_ids} for token_ids in prompt_token_ids],
                sampling_params=sampling_params,
            )
            answer = [tokenizer.apply_chat_template(
                [{
                    "role": "assistant",
                    # Delete the template of Llama
                    "content": o.outputs[0].text.replace('assistant\n\n', '')
                }]
            )[1:] for o in output] # Delete `bos` token
            
            ds = ds.add_column(model_name, answer)
            ds.save_to_disk(HF_HOME + f'/datasets/{name}_{model_name.replace('/', '_')}')

if __name__ == '__main__':
    model_name = "deepseek-ai/DeepSeek-V2-Lite"
    distill_from_llama(model_name)