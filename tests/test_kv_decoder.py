import time

from src.model.model_setup import load_model
from src.inference.kv_decoder import KVDecoder


def main():

    tokenizer, model = load_model()

    decoder = KVDecoder(
        model,
        tokenizer
    )

    prompt = "Explain how a CPU executes an instruction."

    start = time.perf_counter()

    generated_text = decoder.generate(
        prompt,
        max_new_tokens=50
    )

    end = time.perf_counter()

    latency = end - start

    print("\n--- MANUAL KV DECODER ---")

    print("Prompt:")
    print(prompt)

    print("\nGenerated:")
    print(generated_text)

    print("\nLatency:")
    print(round(latency, 4), "seconds")


if __name__ == "__main__":
    main()