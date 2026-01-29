import json
import os
import re
import requests
import subprocess
import time
from tqdm import tqdm

from benchmarks.benchmark_const import CHAT_TEMPLATE_PREFIX
from data import TEST_DATASETS

def kill_process_tree(pid: int):
    try:
        subprocess.run(['pkill', '-P', str(pid)], check=False)
        subprocess.run(['kill', str(pid)], check=False)
    except Exception as e:
        print(f"Error using shell kill: {e}")

def run_block_attention(model_name, dataset):
    template_prefix = CHAT_TEMPLATE_PREFIX[model_name]
    result_cache_file = f"{dataset.name}.json"
    if os.path.exists(result_cache_file):
        print("Result exists:", result_cache_file)
        return
        
    ds = dataset(model_name, split="test")
    result = ds.data.remove_columns(
        ['normal_input', 'input_ids', 'normal_input_new', 'input_ids_new',
            'chunk_ids', 'cross_attention_mask', 'labels']
    )
    answer = []
    for example in tqdm(result):
        blocks = ["<|user|>\n" + ds.instruction + example['input']]
        delimiter = r'(Dialogue:)' if ds.name == 'samsum' else r'(Passage \d+:\n)' 
        docs = re.split(delimiter, example['context'])
        assert docs[0] == '', f'First element of docs is not empty: {docs[0]}'
        docs = docs[1:]
        blocks.extend([docs[i] + docs[i+1] for i in range(0, len(docs), 2)])
        blocks[-1] += '<|assistant|>\n'
        r = requests.post(
            url="http://localhost:9000/generate",
            data=json.dumps({"blocks": blocks}),
            headers={"Content-Type": "application/json"}
        )
        answer.append(r.json()['generated'].replace(template_prefix, ""))

    result = result.add_column('output_blockattention', answer)
    result = result.map(dataset.scorer, fn_kwargs={'method': 'blockattention'})
    result.to_json(result_cache_file)
    
def run_blockattention_backend() -> subprocess.Popen:
    return subprocess.Popen(["python", "block_generate_server.py",
                    "--port", '9000',
                    "--model", 'ldsjmdy/Tulu3-Block-FT',
                    "--dtype", "bfloat16",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    )

def wait_server_setup(port: int):
    url = f"http://localhost:{port}/health"
    # Wait for dataset preparation and engine initialzation.
    time.sleep(10)
    while True:
        try:
            time.sleep(5)
            response = requests.get(url)
            if response.status_code == 200:
                break
        except Exception as e:
            print(e)

if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    handle = run_blockattention_backend()
    try:
        wait_server_setup(9000)
        print("Server up.")
        for dataset_name, dataset in TEST_DATASETS.items():
            print(f"Running Block-Attention on dataset: {dataset_name}")
            run_block_attention(model_name, dataset)
    finally:
        # Kill the server to save the time of aborting unfinished requests.
        kill_process_tree(handle.pid)
        handle.wait()