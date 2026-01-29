from transformers import AutoTokenizer

from comb import COMB

def main():
    request = {
        "question": "Answer the question based on the given passages. "
            "Only give me the answer and do not output any other words.\n\n"
            "Question: What is the best thing to do in San Francisco?",
        "context": "The best thing to do in San Francisco is "
            "to eat a sandwich and sit in Dolores Park on a sunny day.",
    }

    # For simplicity, we tokenize the request outside of COMB here.
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt = {
        "input_ids": tokenizer.apply_chat_template(
            [{
                "role": "user",
                "content": request["question"]
            }]
        ),
        "chunk_ids": tokenizer(request["context"])["input_ids"],
    }

    # Create COMB instance
    comb = COMB(model_name)

    # Set global sampling parameters, or override in generate() call
    comb.set_sampling_params(temperature=0.0)

    # Generate answer
    output = comb.generate(prompt)[0]
    answer = tokenizer.decode(output.token_ids, skip_special_tokens=True)
    print("INPUT: \n", request["question"])
    print("CONTEXT: \n", request["context"])
    # Eat a sandwich and sit in Dolores Park on a sunny day.
    print("ANSWER: \n", answer)

if __name__ == "__main__":
    main()