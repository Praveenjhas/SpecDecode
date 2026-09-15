import time

from src.model.model_setup import load_model
from src.inference.engine import InferenceEngine


def main():

    tokenizer, model = load_model()

    engine = InferenceEngine(
        model=model,
        tokenizer=tokenizer,
        max_running_requests=4,
        prefix_block_size=4,
        prefix_max_blocks=64
    )

    prompt = "Explain what a CPU is."

    print("\n--- ENGINE TEST ---")

    start = time.perf_counter()

    output = engine.generate(
        prompt=prompt,
        max_new_tokens=20
    )

    # synchronize so timing includes GPU work
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end = time.perf_counter()

    print("\nPrompt:")
    print(prompt)

    print("\nGenerated:")
    print(output)

    print("\nLatency:")
    print(
        round(end - start, 4),
        "seconds"
    )

    print("\nPrefix cache stats:")
    print(
        engine.cache_stats()
    )


if __name__ == "__main__":
    main()