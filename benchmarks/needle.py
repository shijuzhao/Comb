from datasets import Dataset
from math import floor
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
import os
from transformers import AutoTokenizer
from vllm import LLM

from benchmark_const import CHAT_TEMPLATE_PREFIX
from comb import COMB
from data.needle import NeedleDataset

def needle_in_a_haystack(model_name, plot=False):
    result_folder = "results/needle/"
    file_name = model_name.replace('/', '_')
    result_cache_file = result_folder + file_name + ".json"
    ds = NeedleDataset(model_name, split="test")
    if os.path.exists(result_cache_file):
        print("Result exists:", result_cache_file)
        result = Dataset.from_json(result_cache_file)
    else:
        result = run_and_score(model_name, ds.data)
        # Remove token ids to save space
        result = result.remove_columns(
            ['normal_input', 'input_ids', 'chunk_ids', 'cross_attention_mask', 'labels']
        )
        result.to_json(result_cache_file)
        
    if plot:
        x_labels = [str(x // 1000) + 'K' for x in ds.context_lengths]
        y_labels = [floor(y) for y in ds.document_depth_percents]
        m, n = len(x_labels), len(y_labels)
        fontsize = 20
        # Create colorbar separately
        fig_cbar = plt.figure(figsize=(0.4, 4))
        ax_cbar = fig_cbar.add_axes([0, 0, 1, 1])
        ColorbarBase(ax_cbar, cmap='YlGn', norm=Normalize(vmin=0, vmax=1), 
                    orientation='vertical')
        ax_cbar.set_ylabel('Accuracy', rotation=-90, va="bottom", fontsize=fontsize)
        fig_cbar.savefig(f"{result_folder}{file_name}_colorbar.pdf", bbox_inches='tight')
        plt.close(fig_cbar)
        for method in ['normal', 'comb']:
            fig, ax = plt.subplots()
            score = [[result[j*n+i][f'score_{method}'] for j in range(m)] for i in range(n)]
            ax.imshow(score, cmap='YlGn', vmin=0, vmax=1)
            ax.set_xticks(range(m))
            ax.set_yticks(range(n))
            ax.set_xticklabels(x_labels)
            ax.set_yticklabels(y_labels)
            ax.set_xlabel('Context Length (tokens)', fontsize=fontsize)
            ax.set_ylabel('Depth Percent (%)', fontsize=fontsize)
            plt.savefig(f"{result_folder}{method}_{file_name}.pdf", bbox_inches='tight')
            plt.close(fig)

def run_and_score(model_name, data):
    template_prefix = CHAT_TEMPLATE_PREFIX[model_name]
    llm = LLM(model=model_name)
    sampling_params = llm.get_default_sampling_params()
    sampling_params.max_tokens = 512
    sampling_params.temperature = 0
    output = llm.generate(
        [{"prompt_token_ids": prompt} for prompt in data["normal_input"]],
        sampling_params=sampling_params,
    )
    answer = [o.outputs[0].text.replace(template_prefix, "") for o in output]
    data = data.add_column('output_normal', answer)
    data = data.map(NeedleDataset.scorer, fn_kwargs={'method': 'normal'})
    del llm

    comb = COMB(model_name, pbc_memory_utilization=0.5)
    comb.set_sampling_params(temperature=0.0, max_tokens=512)
    output = comb.generate(
        data,
        need_store=False,   # We do not need to reuse PIC here.
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    answer = [tokenizer.decode(o.token_ids, skip_special_tokens=True
                    ).replace(template_prefix, "") for o in output]
    data = data.add_column('output_comb', answer)
    data = data.map(NeedleDataset.scorer, fn_kwargs={'method': 'comb'})
    
    return data