import argparse
from datasets import Dataset
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from transformers import AutoTokenizer
from vllm import LLM

from benchmark_const import CHAT_TEMPLATE_PREFIX
from comb import COMB
from data import TEST_DATASETS, DATASET_NAME_PROJ
from data.metrics import qa_f1_score

def offline_experiment(model_name, dataset):
    result_folder = "results/offline/" + model_name.replace('/', '_')
    template_prefix = CHAT_TEMPLATE_PREFIX[model_name]
    result_cache_file = result_folder + f"/{dataset.name}.json"
    if os.path.exists(result_cache_file):
        print("Result exists:", result_cache_file)
    else:
        # Reduce the output length to save time if the metric is f1 score.
        # When the model cannot answer the question, it may repeat a word.
        max_len = 128 if dataset.metric == qa_f1_score else 4096
        ds = dataset(model_name, split="test")
        data = run_vllm(model_name, ds.data, template_prefix, max_len)
        data = data.map(dataset.scorer, fn_kwargs={'method': 'normal'})
        data = run_comb(model_name, data, template_prefix, max_len)
        data = data.map(dataset.scorer, fn_kwargs={'method': 'comb'})
        # Remove token ids to save space
        result = data.remove_columns(
            ['normal_input', 'input_ids', 'normal_input_new', 'input_ids_new',
             'chunk_ids', 'cross_attention_mask', 'labels']
        )
        result.to_json(result_cache_file)

def plot_score(model_name):
    # We use a bar chart to show F1 score, because most F1 scores are 0 or 1.
    # We use a boxplot to show RougeL score.
    f1_score_dataset = ['hotpotqa', 'musique', '2wikimqa']
    rougel_dataset = ['multi_news', 'samsum']
    x = np.arange(len(f1_score_dataset))
    total_width, n = 0.72, 2
    width = total_width / n
    color_list = ['#BC3D27','#0271B4','#DF862B','#228350','#7776B1','#925f36','#2d2f2f','#eecabc']
    hatch = ['xxxx','//','\\\\','++']
    fig, ax = plt.subplots()
    fig.set_size_inches(3.3, 3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    methods = ['normal', 'comb']
    acc_dict = {method: [] for method in methods}
    for dataset in f1_score_dataset:
        result_folder = "results/offline/" + model_name.replace('/', '_')
        result_cache_file = result_folder + f"/{dataset}.json"
        if not os.path.exists(result_cache_file):
            print(f"Warning: {result_cache_file} not found.")
            continue

        result = Dataset.from_json(result_cache_file)
        for method in methods:
            acc_dict[method].append(np.mean(result[f'score_{method}']))

    bars = []
    ax_text_fsize = 9
    for i in range(n):
        bar = ax.bar(x + i * width, acc_dict[methods[i]], edgecolor=color_list[i], fill=False,
                    alpha=1, hatch=hatch[i], width=width, linewidth=1.5, label=methods[i])
        bars.append(bar)
        for j in range(len(x)):
            text_str = str(acc_dict[methods[i]][j])[2:4].lstrip('0')
            ax.text(x[j] + i * width, 1.04 * acc_dict[methods[i]][j], text_str,
                    fontsize=ax_text_fsize, horizontalalignment='center')

    fontsize = 12
    ax.set_ylabel("F1 Score", fontsize=fontsize)
    xtick_label = [DATASET_NAME_PROJ[ds] for ds in f1_score_dataset]
    ax.set_xticks(x + ((n-1)/2) * width, xtick_label)
    ax.tick_params(labelsize=fontsize)
    ax.set_ylim([0.0, 0.6])
    fig.savefig(f"{result_folder}/fig-f1-score.pdf", bbox_inches='tight')
    plt.close()
    legend_fig = plt.figure("legend plot",figsize=(4, 0.8))
    legend_fig.legend(bars, ['Prefix caching', 'COMB'], mode="expand", ncol=2, loc='center', frameon=False)
    legend_fig.savefig(f"{result_folder}/fig-legend.pdf", bbox_inches='tight')

    capprops = dict(linewidth=0) 
    medianprops = dict(linestyle = '--', color='black')
    fig, ax = plt.subplots()
    fig.set_size_inches(2.2, 3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    size = 3
    xtick_list = []
    xtick_label_list = []
    for i, dataset in enumerate(rougel_dataset):
        result_folder = "results/offline/" + model_name.replace('/', '_')
        result_cache_file = result_folder + f"/{dataset}.json"
        if not os.path.exists(result_cache_file):
            print(f"Warning: {result_cache_file} not found.")
            continue

        result = Dataset.from_json(result_cache_file)
        method_place_list = []
        for j, method in enumerate(methods):
            method_place_list.append(i * size + 0.5 * j)
            boxprops1 = dict(linestyle='-', linewidth=1.5, color=color_list[j], facecolor='none', hatch=hatch[j])
            whiskerprops1 = dict(linestyle='-', linewidth=1, color=color_list[j])
            ax.boxplot(result[f'score_{method}'], positions=[i * size + 0.5 * j],
                patch_artist=True, boxprops=boxprops1, whiskerprops=whiskerprops1,
                whis=[5, 95], capprops=capprops, medianprops=medianprops,
                widths=0.4, showfliers=False)

        xtick_list.append(sum(method_place_list)/len(method_place_list))
        xtick_label_list.append(DATASET_NAME_PROJ[dataset])
        
    plt.xticks(xtick_list, xtick_label_list, fontsize = 12)
    plt.ylabel('Rouge-L Score',fontsize = 12)
    plt.tight_layout()
    plt.savefig(f"{result_folder}/fig-RougeL.pdf", bbox_inches='tight')
    plt.close(fig)

def plot_ttft(model_name, metric):
    color_list = ['#BC3D27','#0271B4','#DF862B','#228350','#7776B1','#925f36','#2d2f2f','#eecabc']
    hatch = ['xxxx','//','\\\\','++']
    capprops = dict(linewidth=0) 
    medianprops = dict(linestyle = '--', color='black')
    fig, ax = plt.subplots()
    fig.set_size_inches(5.5, 3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    size = 3
    xtick_list = []
    xtick_label_list = []
    for i, (dataset, xtick_label) in enumerate(DATASET_NAME_PROJ.items()):
        result_folder = "results/offline/" + model_name.replace('/', '_')
        result_cache_file = result_folder + f"/{dataset}.json"
        if not os.path.exists(result_cache_file):
            print(f"Warning: {result_cache_file} not found.")
            continue

        result = Dataset.from_json(result_cache_file)
        method_place_list = []
        for j, method in enumerate(['normal', 'comb']):
            method_place_list.append(i * size + 0.5 * j)
            boxprops1 = dict(linestyle='-', linewidth=1.5, color=color_list[j], facecolor='none', hatch=hatch[j])
            whiskerprops1 = dict(linestyle='-', linewidth=1, color=color_list[j])
            ax.boxplot(result[f'{metric}_{method}'], positions=[i * size + 0.5 * j],
                patch_artist=True, boxprops=boxprops1, whiskerprops=whiskerprops1,
                whis=[5, 95], capprops=capprops, medianprops=medianprops,
                widths=0.4, showfliers=False)

        xtick_list.append(sum(method_place_list)/len(method_place_list))
        xtick_label_list.append(xtick_label)
        
    plt.xticks(xtick_list, xtick_label_list, fontsize = 12)
    plt.ylabel('TTFT (s)',fontsize = 12)
    plt.tight_layout()
    plt.savefig(f"{result_folder}/fig-{metric}.pdf", bbox_inches='tight')
    plt.close(fig)

def run_block_attention(model_name, dataset):
    result_folder = "results/offline/" + model_name.replace('/', '_')
    template_prefix = CHAT_TEMPLATE_PREFIX[model_name]
    result_cache_file = result_folder + f"/{dataset.name}.json"
    if os.path.exists(result_cache_file):
        print("Result exists:", result_cache_file)
        result = Dataset.from_json(result_cache_file)
        if 'score_blockattention' in result.column_names:
            print("Block attention score exists.")
            return
    else:
        raise ValueError("Result file does not exist. "
                    "Please run offline_experiment first.")
        
    ds = dataset(model_name, split="test")

    # Reduce the output length to save time if the metric is f1 score.
    # When the model cannot answer the question, it may repeat a word.
    max_len = 128 if dataset.metric == qa_f1_score else 4096
    llm = LLM(model='ldsjmdy/Tulu3-Block-FT', disable_log_stats=False)
    sampling_params = llm.get_default_sampling_params()
    sampling_params.max_tokens = max_len
    sampling_params.temperature = 0
    answer = []
    for prompt in ds["normal_input"]:
        output = llm.generate(
            {"prompt_token_ids": prompt},
            sampling_params=sampling_params,
            use_tqdm=False,
        )[0]
        answer.append(output.outputs[0].text.replace(template_prefix, ""))

    del llm
    result = result.add_column('output_blockattention', answer)
    result = result.map(dataset.scorer, fn_kwargs={'method': 'blockattention'})
    result.to_json(result_cache_file)

def run_comb(model_name, data, template_prefix, max_len):
    comb = COMB(model_name, disable_log_stats=False, pbc_memory_utilization=0.5)
    comb.set_sampling_params(temperature=0.0, max_tokens=max_len)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    answer, ttft1, ttft2 = [], [], []
    for prompt in data:
        output = comb.generate(prompt)[0]
        answer.append(tokenizer.decode(output.token_ids,
                        skip_special_tokens=True).replace(template_prefix, ""))
        ttft1.append(output.first_token_latency)
        prompt["input_ids"] = prompt.pop("input_ids_new")
        output = comb.generate(prompt, max_tokens=1)[0]
        ttft2.append(output.first_token_latency)

    data = data.add_column('output_comb', answer)
    data = data.add_column('ttft1_comb', ttft1)
    data = data.add_column('ttft2_comb', ttft2)
    return data

def run_vllm(model_name, data, template_prefix, max_len):
    llm = LLM(model=model_name, disable_log_stats=False)
    sampling_params1 = llm.get_default_sampling_params()
    sampling_params1.max_tokens = max_len
    sampling_params1.temperature = 0
    sampling_params2 = sampling_params1.clone()
    # We set the second output length as 1 to save time
    sampling_params2.max_tokens = 1
    answer, ttft1, ttft2 = [], [], []
    for prompt, prompt_new in zip(data["normal_input"], data["normal_input_new"]):
        output = llm.generate(
            {"prompt_token_ids": prompt},
            sampling_params=sampling_params1,
            use_tqdm=False,
        )[0]
        answer.append(output.outputs[0].text.replace(template_prefix, ""))
        ttft1.append(output.metrics.first_token_latency)
        output = llm.generate(
            {"prompt_token_ids": prompt_new},
            sampling_params=sampling_params2,
            use_tqdm=False,
        )[0]
        ttft2.append(output.metrics.first_token_latency)

    del llm
    data = data.add_column('output_normal', answer)
    data = data.add_column('ttft1_normal', ttft1)
    data = data.add_column('ttft2_normal', ttft2)
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--block_attention", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    dataset = TEST_DATASETS[args.dataset]
    if args.block_attention:
        run_block_attention(args.model, dataset)
    else:
        offline_experiment(args.model, dataset)