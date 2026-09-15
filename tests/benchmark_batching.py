import time

import torch

from src.model.model_setup import load_model
from src.inference.kv_decoder import KVDecoder
from src.inference.batched_decoder import BatchedDecoder


def synchronize():
    """
    CUDA operations are asynchronous, so synchronize before
    reading wall-clock timings.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def percentile(values, p):
    values = sorted(values)

    index = int((p / 100) * (len(values) - 1))

    return values[index]


def main():

    tokenizer, model = load_model()

    decoder = KVDecoder(
        model,
        tokenizer
    )

    batched_decoder = BatchedDecoder(
        model,
        tokenizer
    )

    prompts = [
        "Explain what a CPU is.",
        "Explain what a CPU is.",
        "Explain what a CPU is.",
        "Explain what a CPU is.",
    ]

    num_requests = len(prompts)

    print("\n--- PREPARING KV CACHES ---")

    tokens = []
    caches = []

    for prompt in prompts:

        input_ids, next_token, cache = (
            decoder.prefill(prompt)
        )

        tokens.append(
            next_token.item()
        )

        caches.append(cache)

    kv_length = caches[0].layers[0].keys.shape[2]

    print("Requests:", num_requests)
    print("KV length:", kv_length)

    # -------------------------------------------------
    # Warmup
    # -------------------------------------------------

    print("\n--- WARMUP ---")

    for _ in range(3):

        for i in range(num_requests):

            decoder.step(
                torch.tensor(
                    tokens[i],
                    device="cuda"
                ),
                caches[i]
            )

        batched_decoder.decode_batch(
            tokens,
            caches
        )

    synchronize()

    # -------------------------------------------------
    # Individual benchmark
    # -------------------------------------------------

    individual_times = []

    print("\n--- INDIVIDUAL DECODE BENCHMARK ---")

    for _ in range(10):

        synchronize()

        start = time.perf_counter()

        for i in range(num_requests):

            decoder.step(
                torch.tensor(
                    tokens[i],
                    device="cuda"
                ),
                caches[i]
            )

        synchronize()

        end = time.perf_counter()

        individual_times.append(
            end - start
        )

    # -------------------------------------------------
    # Batched benchmark
    # -------------------------------------------------

    batched_times = []

    print("\n--- BATCHED DECODE BENCHMARK ---")

    for _ in range(10):

        synchronize()

        start = time.perf_counter()

        batched_decoder.decode_batch(
            tokens,
            caches
        )

        synchronize()

        end = time.perf_counter()

        batched_times.append(
            end - start
        )

    # -------------------------------------------------
    # Results
    # -------------------------------------------------

    individual_avg = (
        sum(individual_times)
        / len(individual_times)
    )

    batched_avg = (
        sum(batched_times)
        / len(batched_times)
    )

    speedup = (
        individual_avg
        / batched_avg
    )

    print("\n--- RESULTS ---")

    print(
        "Individual average:",
        round(individual_avg * 1000, 3),
        "ms"
    )

    print(
        "Individual p50:",
        round(
            percentile(individual_times, 50) * 1000,
            3
        ),
        "ms"
    )

    print(
        "Individual p95:",
        round(
            percentile(individual_times, 95) * 1000,
            3
        ),
        "ms"
    )

    print(
        "Batched average:",
        round(batched_avg * 1000, 3),
        "ms"
    )

    print(
        "Batched p50:",
        round(
            percentile(batched_times, 50) * 1000,
            3
        ),
        "ms"
    )

    print(
        "Batched p95:",
        round(
            percentile(batched_times, 95) * 1000,
            3
        ),
        "ms"
    )

    print(
        "Speedup:",
        round(speedup, 2),
        "x"
    )

    print(
        "Requests per batch:",
        num_requests
    )


if __name__ == "__main__":
    main()