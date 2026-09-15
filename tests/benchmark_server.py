import concurrent.futures
import statistics
import time

import requests


URL = "http://127.0.0.1:8000/generate/batched"

PROMPTS = [
    "Explain what a CPU is.",
    "Explain what RAM is.",
    "Explain what a GPU is.",
    "Explain what an SSD is.",
    "Explain what a process is.",
    "Explain what a thread is.",
    "Explain what virtual memory is.",
    "Explain what a database is.",
]


TOTAL_REQUESTS = 20
CONCURRENCIES = [1, 2, 4, 8]


def percentile(values, p):
    values = sorted(values)

    if not values:
        return 0.0

    position = (len(values) - 1) * p / 100

    lower = int(position)
    upper = min(lower + 1, len(values) - 1)

    if lower == upper:
        return values[lower]

    fraction = position - lower

    return (
        values[lower]
        + fraction * (values[upper] - values[lower])
    )


def send_request(index):
    prompt = PROMPTS[index % len(PROMPTS)]

    body = {
        "prompt": prompt,
        "max_new_tokens": 20
    }

    start = time.perf_counter()

    response = requests.post(
        URL,
        json=body,
        timeout=180
    )

    end = time.perf_counter()

    response.raise_for_status()

    result = response.json()

    return {
        "request_id": result["request_id"],
        "latency": end - start,
        "tokens": result["generated_tokens"]
    }


def run_load(concurrency):
    start = time.perf_counter()

    results = []

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=concurrency
    ) as executor:

        futures = [
            executor.submit(
                send_request,
                i
            )
            for i in range(TOTAL_REQUESTS)
        ]

        for future in concurrent.futures.as_completed(
            futures
        ):
            results.append(
                future.result()
            )

    wall_time = time.perf_counter() - start

    return results, wall_time


def main():

    # ---------------------------------------------
    # Warm-up
    # ---------------------------------------------

    print("\n--- WARM-UP ---")

    for i in range(3):
        send_request(i)

    print(
        f"\nRunning {TOTAL_REQUESTS} requests "
        f"per concurrency level."
    )

    print(
        "Concurrency levels:",
        CONCURRENCIES
    )

    # ---------------------------------------------
    # Benchmark
    # ---------------------------------------------

    for concurrency in CONCURRENCIES:

        print(
            f"\n{'=' * 55}"
        )

        print(
            f"CONCURRENCY = {concurrency}"
        )

        print(
            f"{'=' * 55}"
        )

        results, wall_time = run_load(
            concurrency
        )

        latencies = [
            result["latency"]
            for result in results
        ]

        total_tokens = sum(
            result["tokens"]
            for result in results
        )

        mean_latency = statistics.mean(
            latencies
        )

        p50 = percentile(
            latencies,
            50
        )

        p95 = percentile(
            latencies,
            95
        )

        p99 = percentile(
            latencies,
            99
        )

        requests_per_second = (
            len(results) / wall_time
        )

        tokens_per_second = (
            total_tokens / wall_time
        )

        print(
            "Completed requests:",
            len(results)
        )

        print(
            "Total wall time:",
            round(wall_time, 4),
            "s"
        )

        print(
            "Mean latency:",
            round(mean_latency, 4),
            "s"
        )

        print(
            "p50 latency:",
            round(p50, 4),
            "s"
        )

        print(
            "p95 latency:",
            round(p95, 4),
            "s"
        )

        print(
            "p99 latency:",
            round(p99, 4),
            "s"
        )

        print(
            "Requests/sec:",
            round(requests_per_second, 3)
        )

        print(
            "Tokens/sec:",
            round(tokens_per_second, 3)
        )

        print(
            "Min latency:",
            round(min(latencies), 4),
            "s"
        )

        print(
            "Max latency:",
            round(max(latencies), 4),
            "s"
        )


if __name__ == "__main__":
    main()