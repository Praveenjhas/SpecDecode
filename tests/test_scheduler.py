from src.model.model_setup import load_model
from src.inference.kv_decoder import KVDecoder
from src.inference.batched_decoder import BatchedDecoder
from src.inference.request import RequestState
from src.scheduler.scheduler import Scheduler


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

    scheduler = Scheduler(
        decoder=decoder,
        batched_decoder=batched_decoder,
        max_running_requests=4
    )

    requests = [
        RequestState(
            "A",
            "Explain what a CPU is.",
            max_new_tokens=5
        ),
        RequestState(
            "B",
            "Explain what a CPU is.",
            max_new_tokens=5
        ),
        RequestState(
            "C",
            "Explain what a GPU is.",
            max_new_tokens=5
        ),
        RequestState(
            "D",
            "Explain what a GPU is.",
            max_new_tokens=5
        ),
    ]

    for request in requests:
        scheduler.add_request(request)

    print("\n--- BATCHED SCHEDULER ---")

    step = 0

    while scheduler.has_work():

        step += 1

        processed = scheduler.step()

        print(
            f"\nScheduler step {step}"
        )

        print(
            "Processed:",
            [r.request_id for r in processed]
        )

        print(
            "Running:",
            [r.request_id for r in scheduler.running]
        )

        print(
            "Waiting:",
            [r.request_id for r in scheduler.waiting]
        )

        for request in processed:

            token = tokenizer.decode(
                [request.current_token.item()]
            )

            print(
                f"  {request.request_id}"
                f" -> {token!r}"
            )

    print("\n--- FINAL RESULTS ---")

    for request in requests:

        text = tokenizer.decode(
            request.generated_tokens,
            skip_special_tokens=True
        )

        print(
            f"{request.request_id}: "
            f"reason={request.finish_reason}"
        )

        print(
            f"  Generated: {text!r}"
        )


if __name__ == "__main__":
    main()