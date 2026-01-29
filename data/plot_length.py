import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

from data import TEST_DATASETS, DATASET_NAME_PROJ

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.weight'] = 'medium'
plt.rcParams['hatch.linewidth'] = 0.5

model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
input_length = []
output_length = []
dataset_names = []
for name, dataset in TEST_DATASETS.items():
    ds = dataset(model_name, split="test")
    input_length.append([len(example['normal_input']) for example in ds])
    output_length.append([len(tokenizer.encode(example['answers'][0])) 
                            for example in ds])
    dataset_names.append(name)

fontsize = 20
marker_styles = ['o', 's', 'x', '*', '<', '>', 'p', 'h', 'D', 'v', '+', '^']
_, ax = plt.subplots(figsize=(4, 4))
lines = []
for i, name in enumerate(dataset_names):
    line = ax.plot(sorted(input_length[i]), np.arange(0, 1, 1/len(input_length[i])),
                    marker=marker_styles[i], label=name, markevery=15)
    lines.append(line[0])

plt.xlabel("Prompt Length (token)", fontsize=fontsize)
plt.ylabel("CDF", fontsize=fontsize)
plt.tick_params(axis='both', which='major', labelsize=fontsize)
plt.savefig('fig-inputlen.pdf', bbox_inches='tight')
plt.close()

legend_fig = plt.figure("legend plot",figsize=(6, 0.8))
legend_name = [DATASET_NAME_PROJ[name] for name in dataset_names]
legend_fig.legend(lines, legend_name, mode="expand", ncol=5, loc='center', frameon=False)
legend_fig.savefig('fig-len-legend.pdf', bbox_inches='tight')

_, ax = plt.subplots(figsize=(4, 4))
for i, name in enumerate(dataset_names):
    ax.plot(sorted(output_length[i]), np.arange(0, 1, 1/len(output_length[i])),
                    marker=marker_styles[i], label=name, markevery=15)

plt.xlabel("Answer Length (token)", fontsize=fontsize)
plt.ylabel("CDF", fontsize=fontsize)
plt.tick_params(axis='both', which='major', labelsize=fontsize)
plt.savefig('fig-outputlen.pdf', bbox_inches='tight')
plt.close()