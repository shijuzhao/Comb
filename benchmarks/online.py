import asyncio
from collections import defaultdict
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import os
import random
import re
import requests
import subprocess
import sys
import time
from transformers import AutoTokenizer
from typing import List, Dict, Optional

class DummyDataset:
    """A dataset containing random prompts for benchmarking.

    We randomly generate tokens to control the prompt lengths and output lengths.
    """
    def __init__(self, model_name: str):
        num_datapoints = 30
        instruction = np.random.randint(1, 100000, 20).tolist()
        instruction_new = np.random.randint(1, 100000, 20).tolist()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.data = []
        for _ in range(num_datapoints):
            context_len = random.randint(12288, 20480)
            context_ids = np.random.randint(1, 100000, context_len).tolist()
            self.data.append({
                "prompt": tokenizer.decode(instruction + context_ids),
                "prompt_new": tokenizer.decode(instruction_new + context_ids),
                "input_ids": instruction,
                "input_ids_new": instruction_new,
                "chunk_ids": context_ids,
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        return self.data[idx]

def get_vllm_ttft_metrics(
        model_name: str,
        vllm_url: str ="http://localhost:8000"
    ) -> Optional[tuple[float, int]]:
    try:
        metrics_url = f"{vllm_url}/metrics"
        response = requests.get(metrics_url, timeout=10)
        response.raise_for_status()
        metrics_text = response.text
        ttft_sum = None
        ttft_count = None
        sum_pattern = 'vllm:time_to_first_token_seconds_sum{engine="0",' \
                + f'model_name="{model_name}"' + r'} (\d+\.\d+)'
        count_pattern = 'vllm:time_to_first_token_seconds_count{engine="0",' \
                + f'model_name="{model_name}"' + r'} (\d+)'
        sum_match = re.search(sum_pattern, metrics_text)
        count_match = re.search(count_pattern, metrics_text)
        if sum_match:
            ttft_sum = float(sum_match.group(1))
        if count_match:
            ttft_count = int(count_match.group(1))
        
        return (ttft_sum, ttft_count)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching metrics: {e}")
        return None
    except Exception as e:
        print(f"Error parsing metrics: {e}")
        return None

def kill_process_tree(pid: int):
    try:
        subprocess.run(['pkill', '-P', str(pid)], check=False)
        subprocess.run(['kill', str(pid)], check=False)
    except Exception as e:
        print(f"Error using shell kill: {e}")

def run_comb_backend(model: str, port: int) -> subprocess.Popen:
    return subprocess.Popen(["python3", "-m", "comb.entrypoints.api_server",
                    "--port", f"{port}",
                    "--model", model,
                    "--pic-memory-utilization", "0.5",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=sys.stdout,
                    )

def run_vllm_backend(model: str, port: int) -> subprocess.Popen:
    return subprocess.Popen(["python3", "-m", "vllm.entrypoints.openai.api_server",
                    "--port", f"{port}",
                    "--model", model,
                    # "--enforce-eager"
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    )

def wait_server_setup(port: int):
    url = f"http://localhost:{port}/health"
    # Wait for dataset preparation and engine initialzation.
    time.sleep(30)
    while True:
        try:
            time.sleep(5)
            response = requests.get(url)
            if response.status_code == 200:
                break
        except Exception as e:
            print(e)
        
class ChatClient:
    def __init__(self, model: str, port: int, method: str, req_rate: float):
        self.time_start: float = 0 # set in run
        self.time_stop: Optional[float] = None # set in run
        self.req_list: List[Dict] = []
        self.model = model
        self.base_url = f"http://localhost:{port}"
        self.method = method
        self.req_rate = req_rate
        self.tot_time = 40

    async def handle_main_request(self, prompt: Dict):
        if self.method == 'normal':
            res = await self.query_vllm(prompt["prompt_new"])
        else:
            prompt_new = {
                "input_ids": prompt["input_ids_new"],
                "chunk_ids": prompt["chunk_ids"]
            }
            res = await self.query_comb(prompt_new)
            
        if self.time_stop is None: # haven't finished yet
            self.req_list.append(res)

    async def query_comb(self,
        prompt: Dict,
        max_tokens: int = 16
    ) -> Dict:
        response = requests.post(
            f"{self.base_url}/generate",
            json={"prompt": prompt, "max_tokens": max_tokens},
            timeout=self.tot_time,
        )
        return response.json()

    async def query_vllm(self,
        prompt: str,
        max_tokens: int = 16
    ) -> Dict:
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{
                    "role": "user",
                    "content": prompt
                }],
                "max_tokens": max_tokens
            },
            timeout=self.tot_time,
        )
        return response.json()
    
    async def run(self, prompts: DummyDataset):
        self.num_datapoints = len(prompts)
        print("Experiment begins.")
        # First few sessions.
        num_first_few_sessions = math.ceil(self.req_rate)
        coroutines = [
            asyncio.create_task(
                self.handle_main_request(prompts[i % self.num_datapoints])
            ) for i in range(num_first_few_sessions)
        ]

        # Start the timer.
        self.time_start = time.perf_counter()

        # Start the first few sessions.
        now_id = num_first_few_sessions
        tot_req = self.tot_time * self.req_rate
        while now_id < tot_req:
            inter_arrival = random.expovariate(self.req_rate)
            await asyncio.sleep(inter_arrival)
            
            coroutines.append(asyncio.create_task(
                self.handle_main_request(prompts[now_id % self.num_datapoints])
            ))
            now_id += 1

        _, pending = await asyncio.wait(coroutines, timeout=30.0)
        self.time_stop = time.perf_counter()

        # Cancel all remaining tasks.
        for task in pending:
            if not task.done():
                task.cancel()

        await asyncio.wait(coroutines, timeout=1.0)

    def data_analysis(self, prev_ttft_sum: float, prev_ttft_count: int):
        num_prompt_tokens = 0
        num_completion_tokens = 0
        if self.method == 'normal':
            ttft_sum, ttft_count = get_vllm_ttft_metrics(self.model, self.base_url)
            ave_ttft = (ttft_sum-prev_ttft_sum) / (ttft_count-prev_ttft_count)
            for req in self.req_list:
                num_prompt_tokens += req['usage']['prompt_tokens']
                num_completion_tokens += req['usage']['completion_tokens']
        else:
            ttft = []
            for req in self.req_list:
                ttft.append(req['first_token_latency'])
                num_prompt_tokens += req['num_prompt_tokens']
                num_completion_tokens += req['num_generation_tokens']

            ave_ttft = np.mean(ttft)

        result_folder = "results/online"
        if not os.path.exists(result_folder):
            os.mkdir(result_folder)

        result_cache_file = result_folder + "/throughput.csv"
        if not os.path.exists(result_cache_file):
            with open(result_cache_file, "a") as f:
                f.write("method,req_rate,num_datapoints,ave_ttft,time_elapsed,"
                        "req_served,num_prompt_tokens,num_completion_tokens\n")

        with open(result_cache_file, "a") as f:
            f.write(f"{self.method},{self.req_rate},{self.num_datapoints},"
                f"{ave_ttft},{self.time_stop - self.time_start},{len(self.req_list)},"
                f"{num_prompt_tokens},{num_completion_tokens}\n")

async def run_client(method: str, model: str, port: int, ds: DummyDataset, req_rate: float):
    client = ChatClient(model, port, method, req_rate)
    num_datapoints = len(ds)
    # Simulate previous queries before experiment.
    print("Simulating previous queries...")
    for i in range(num_datapoints - 1, -1, -1):
        if method == 'normal':
            await client.query_vllm(ds[i]["prompt"], max_tokens=1)
        else:
            prompt = {
                "input_ids": ds[i]["input_ids"],
                "chunk_ids": ds[i]["chunk_ids"]
            }
            await client.query_comb(prompt, max_tokens=1)
    
    # Record the average TTFT of previous queries.
    ttft_sum, ttft_count = None, None
    if method == 'normal':
        ttft_sum, ttft_count = get_vllm_ttft_metrics(model, client.base_url)
    await client.run(ds)
    client.data_analysis(ttft_sum, ttft_count)

def online_experiment(model_name: str, port: int, dataset: DummyDataset, req_rate: float):
    t1 = time.time()
    print("Req rate: ", req_rate)
    print("Start benchmarking vLLM.")
    handle = run_vllm_backend(model_name, port)
    try:
        wait_server_setup(port)
        print("Server up.")
        asyncio.run(run_client('normal', model_name, port, dataset, req_rate))
    finally:
        # Kill the server to save the time of aborting unfinished requests.
        kill_process_tree(handle.pid)
        handle.wait()

    t2 = time.time()
    print(t2 - t1)
    print("Start benchmarking COMB.")
    handle = run_comb_backend(model_name, port)
    try:
        wait_server_setup(port)
        print("Server up.")
        asyncio.run(run_client('comb', model_name, port, dataset, req_rate))
    finally:
        # Kill the server to save the time of aborting unfinished requests.
        kill_process_tree(handle.pid)
        handle.wait()

    print(time.time() - t2)

def plot_results():
    results_folder = "results/online/"
    results = np.loadtxt(results_folder + 'throughput.csv', delimiter=',', skiprows=1,
                         dtype={'names': ('method', 'req_rate', 'num_datapoints',
                                          'ave_ttft', 'time_elapsed',
                                          'req_served', 'num_prompt_tokens',
                                          'num_completion_tokens'),
                        'formats': ('U10', 'f4', 'i4', 'f8', 'f8', 'i4', 'i8', 'i8')})
    
    methods = ['normal', 'cacheblend', 'kvlink', 'comb']
    ttft = {method: defaultdict(float) for method in methods}
    throughput = {method: defaultdict(float) for method in methods}
    for row in results:
        method = row['method']
        req_rate = row['req_rate']
        ttft[method][req_rate] = row['ave_ttft']
        throughput[method][req_rate] = row['num_prompt_tokens'] / row['time_elapsed']

    fontsize = 20
    marker_styles = ['o', 's', 'x', '*', '<', '>', 'p', 'h', 'D', 'v', '+', '^']
    _, ax = plt.subplots(figsize=(4, 4))
    lines = []
    for i, method in enumerate(ttft.keys()):
        req_rates = sorted(ttft[method].keys())
        ttft_values = [ttft[method][r] for r in req_rates]
        line = ax.plot(req_rates, ttft_values, marker=marker_styles[i],
                 label=method)
        lines.append(line[0])

    plt.xlabel("Request Rate (req/s)", fontsize=fontsize)
    plt.ylabel("TTFT (s)", fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.savefig(results_folder + 'fig-online-a.pdf', bbox_inches='tight')
    plt.close()

    legend_fig = plt.figure("legend plot",figsize=(6, 0.8))
    legend_fig.legend(lines, ['Prefix Caching', 'CacheBlend', 'EPIC', 'COMB'], mode="expand", ncol=4, loc='center', frameon=False)
    legend_fig.savefig(results_folder + 'fig-online-legend.pdf', bbox_inches='tight')

    _, ax = plt.subplots(figsize=(4, 4))
    for i, method in enumerate(throughput.keys()):
        req_rates = sorted(throughput[method].keys())
        ttft_values = [throughput[method][r] for r in req_rates]
        line = ax.plot(req_rates, ttft_values, marker=marker_styles[i],
                 label=method)

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 3))
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.get_offset_text().set_position((0, 1))
    plt.xlabel("Request Rate (req/s)", fontsize=fontsize)
    plt.ylabel("Throughput (token/s)", fontsize=fontsize)
    plt.xticks(np.arange(1, max(req_rates)+1))
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.savefig(results_folder + 'fig-online-b.pdf', bbox_inches='tight')