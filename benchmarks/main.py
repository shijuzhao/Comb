#! /usr/bin/env python

"""
This file contains all the functions to do the experiments. 
For convenience, choose the experiment by inputting the arguments, 
and change the parameters in the file 'benchmark_const.py'.
"""

import argparse
import matplotlib
import matplotlib.pyplot as plt
import subprocess
import sys

from benchmark_const import *
from data import TEST_DATASETS

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.weight'] = 'medium'
plt.rcParams['hatch.linewidth'] = 0.5

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="offline",
                        help="Select the experiment to run.")
    parser.add_argument("--plot", action="store_true",
                        help="Whether to plot the results.")

    args = parser.parse_args(sys.argv[1:])

    if args.experiment == "needle":
        from needle import needle_in_a_haystack
        needle_in_a_haystack(MODEL_NAME, plot=args.plot)

    elif args.experiment == "offline":
        for dataset in TEST_DATASETS.keys():
            print("Running experiment on dataset:", dataset)
            # We need to run each experiment in a separate process,
            # because the LLM engine must be restarted before experiments.
            command = f"python offline.py --dataset {dataset} --model {MODEL_NAME}"
            result = subprocess.run(command, shell=True, capture_output=True)
            print(result.stdout.decode())
            print(result.stderr.decode())

        if MODEL_NAME == "meta-llama/Llama-3.1-8B-Instruct":
            command = f"python offline.py --dataset {dataset} --model {MODEL_NAME} --block_attention"
            result = subprocess.run(command, shell=True, capture_output=True)
            print(result.stdout.decode())
            print(result.stderr.decode())

        if args.plot:
            from offline import plot_ttft, plot_score
            plot_ttft(MODEL_NAME, 'ttft1')
            plot_ttft(MODEL_NAME, 'ttft2')
            plot_score(MODEL_NAME)

    elif args.experiment == "online":
        from online import DummyDataset, online_experiment
        model_name = "meta-llama/Llama-3.1-8B-Instruct"
        dataset = DummyDataset(model_name)
        for req_rate in REQ_RATES:
            online_experiment(model_name, PORT, dataset, req_rate)

        if args.plot:
            from online import plot_results
            plot_results()

    elif args.experiment == "cache_size":
        from cache_size import plot_cache_size
        plot_cache_size()