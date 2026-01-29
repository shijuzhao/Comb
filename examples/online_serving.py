import requests
import subprocess
import time
from transformers import AutoTokenizer

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
        except:
            pass

def main(model_name: str, port: int):
    base_url = f"http://localhost:{port}"
    request = {
        "question": "Answer the question based on the given passages. "
            "Only give me the answer and do not output any other words.\n\n"
            "Question: What is the best thing to do in San Francisco?",
        "context": "The best thing to do in San Francisco is "
            "to eat a sandwich and sit in Dolores Park on a sunny day.",
    }

    # For simplicity, we tokenize the request outside of COMB here.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt = {
        "input_ids": tokenizer.apply_chat_template(
            [{
                "role": "user",
                "content": request["question"]
            }]
        ),
        "chunk_ids": tokenizer.encode(request["context"]),
    }

    payload = {"prompt": prompt}
    response = requests.post(
        f"{base_url}/generate",
        json=payload,
        timeout=30,
    )
    output = response.json()
    answer = tokenizer.decode(output["token_ids"], skip_special_tokens=True)
    print("INPUT: \n", request["question"])
    print("CONTEXT: \n", request["context"])
    # Eat a sandwich and sit in Dolores Park on a sunny day.
    print("ANSWER: \n", answer)

if __name__ == '__main__':
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    port = 8080
    handle = run_comb_backend(model_name, port)
    try:
        wait_server_setup(port)
        print("Server up.")
        main(model_name, port)
    finally:
        kill_process_tree(handle.pid)
        pass