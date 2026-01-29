import matplotlib.pyplot as plt
import numpy as np
import os
from transformers import AutoConfig

from comb.supported_models import COMB_MODEL_MAPPING

B_TO_MB = 2**-20

def plot_cache_size():
    methods = ['Prefix caching', 'COMB']
    num_tokens = 1024
    normal_cache_sizes = []
    comb_cache_sizes = []
    for model_name in ["meta-llama/Llama-3.1-8B-Instruct", 
                        "deepseek-ai/DeepSeek-V2-Lite-Chat"]:
        config = AutoConfig.from_pretrained(model_name)
        single_layer_cache_size = (2 * num_tokens * config.num_key_value_heads
                        * config.head_dim * config.dtype.itemsize * B_TO_MB)
        normal_cache_sizes.append(single_layer_cache_size * config.num_hidden_layers)
        comb_config = AutoConfig.from_pretrained(COMB_MODEL_MAPPING[model_name])
        comb_cache_sizes.append(single_layer_cache_size * len(comb_config.cross_attention_layers))

    cache_sizes = [normal_cache_sizes, comb_cache_sizes]
    x = np.arange(len(normal_cache_sizes))
    total_width, n = 0.72, 2
    width = total_width / n
    color_list = ['#BC3D27','#228350',"#0271B4",'#DF862B','#7776B1','#925f36','#2d2f2f']
    hatch = ['xxxx','++','//','\\\\','o']
    fig, ax = plt.subplots()
    fig.set_size_inches(4, 4)

    bars = []
    ax_text_fsize = 10
    for i in range(n):
        bar = ax.bar(x + i * width, cache_sizes[i], edgecolor=color_list[i], fill=False,
                    alpha=1, hatch=hatch[i], width=width, linewidth=1.5, label=methods[i])
        bars.append(bar)
        for j in range(len(x)):
            text_str = str(cache_sizes[i][j]).split('.')[0]
            ax.text(x[j] + i * width, 1.04 * cache_sizes[i][j], text_str,
                    fontsize=ax_text_fsize, horizontalalignment='center')

    fontsize = 12
    ax.set_ylabel("KV Cache Size (MB)", fontsize=fontsize)
    xtick_label = ["Llama-3.1-8B-Instruct", "DeepSeek-V2-Lite-Chat"]
    ax.set_xticks(x + ((n-1)/2) * width, xtick_label)
    ax.tick_params(labelsize=fontsize)
    ax.set_ylim([0.0, 150])
    result_folder = "results/cache_size"
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    fig.savefig(f"{result_folder}/fig-cache-size.pdf", bbox_inches='tight')
    plt.close()
    legend_fig = plt.figure("legend plot",figsize=(3, 0.8))
    legend_fig.legend(bars, methods, mode="expand", ncol=2, loc='center', frameon=False)
    legend_fig.savefig(f"{result_folder}/fig-cache-legend.pdf", bbox_inches='tight')