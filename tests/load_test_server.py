import concurrent.futures
import time

import requests


URL = "http://127.0.0.1:8000/generate/batched"


PROMPTS = [
    "Explain what a CPU is.",
    "Explain what RAM is.",
    "Explain what a GPU is.",
    "Explain what an SSD is.",
]


def send_request(prompt):

    body = {
        "prompt": prompt,
        "max_new_tokens": 20
    }

    start = time.perf_counter()

    response = requests.post(
        URL,
        json=body,
        timeout=120
    )

    end = time.perf_counter()

    response.raise_for_status()

    result = response.json()

    return {
        "request_id": result["request_id"],
        "latency": end - start,
        "generated_tokens":
            result["generated_tokens"]
    }


def main():

    for concurrency in [1, 2, 4]:

        prompts = [
            PROMPTS[i % len(PROMPTS)]
            for i in range(concurrency)
        ]

        print(
            f"\n--- CONCURRENCY {concurrency} ---"
        )

        start = time.perf_counter()

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=concurrency
        ) as executor:

            results = list(
                executor.map(
                    send_request,
                    prompts
                )
            )

        end = time.perf_counter()

        total_time = end - start

        print(
            "Wall time:",
            round(total_time, 4),
            "s"
        )

        for result in results:

            print(
                result["request_id"],
                "latency=",
                round(result["latency"], 4),
                "tokens=",
                result["generated_tokens"]
            )


if __name__ == "__main__":
    main()