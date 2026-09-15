import torch

from src.model.model_setup import load_model
from src.inference.kv_decoder import KVDecoder
from src.inference.batched_decoder import BatchedDecoder


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

    # -------------------------------------------------
    # Candidate prompts
    # -------------------------------------------------

    candidates = [
        "Explain what a CPU is.",
        "Explain what RAM is.",
        "Explain what a GPU is.",
        "Explain what cache memory is.",
        "Explain what virtual memory is.",
    ]

    # Find two prompts with the same token count.
    selected = None

    tokenized = []

    for prompt in candidates:

        ids = tokenizer(
            prompt,
            return_tensors="pt"
        )["input_ids"]

        tokenized.append(
            (prompt, ids)
        )

    for i in range(len(tokenized)):

        for j in range(i + 1, len(tokenized)):

            prompt_a, ids_a = tokenized[i]
            prompt_b, ids_b = tokenized[j]

            if ids_a.shape[1] == ids_b.shape[1]:

                selected = (
                    prompt_a,
                    prompt_b
                )

                break

        if selected is not None:
            break

    if selected is None:
        raise RuntimeError(
            "Could not find two prompts with "
            "the same token count."
        )

    prompts = [
        selected[0],
        selected[1]
    ]

    print("\n--- SELECTED PROMPTS ---")

    for i, prompt in enumerate(prompts):

        ids = tokenizer(
            prompt,
            return_tensors="pt"
        )["input_ids"]

        print(
            f"Request {i}: "
            f"{prompt!r} "
            f"tokens={ids.shape[1]}"
        )

    # -------------------------------------------------
    # Prefill
    # -------------------------------------------------

    print("\n--- PREFILL REQUESTS ---")

    tokens = []
    caches = []

    for i, prompt in enumerate(prompts):

        input_ids, next_token, cache = (
            decoder.prefill(prompt)
        )

        tokens.append(
            next_token.item()
        )

        caches.append(cache)

        print(
            f"Request {i}: "
            f"prompt_tokens={input_ids.shape[1]}"
        )

        print(
            "  KV shape:",
            cache.layers[0].keys.shape
        )

        print(
            "  First generated token:",
            repr(
                tokenizer.decode(
                    [next_token.item()]
                )
            )
        )

    # -------------------------------------------------
    # Verify equal KV lengths
    # -------------------------------------------------

    lengths = [
        cache.layers[0].keys.shape[2]
        for cache in caches
    ]

    print("\nKV lengths:", lengths)

    if len(set(lengths)) != 1:
        raise RuntimeError(
            f"Expected equal KV lengths, got {lengths}"
        )

    # -------------------------------------------------
    # Individual decode
    # -------------------------------------------------

    print("\n--- INDIVIDUAL DECODE ---")

    individual_tokens = []

    for i in range(len(tokens)):

        next_token, _ = decoder.step(
            torch.tensor(
                tokens[i],
                device="cuda"
            ),
            caches[i]
        )

        individual_tokens.append(
            next_token.item()
        )

        print(
            f"Request {i}:",
            repr(
                tokenizer.decode(
                    [next_token.item()]
                )
            )
        )

    # -------------------------------------------------
    # Batched decode
    # -------------------------------------------------

    print("\n--- BATCHED DECODE ---")

    batched_tokens, batched_cache = (
        batched_decoder.decode_batch(
            tokens,
            caches
        )
    )

    for i, token in enumerate(
        batched_tokens.tolist()
    ):

        print(
            f"Request {i}:",
            repr(
                tokenizer.decode([token])
            )
        )

    # -------------------------------------------------
    # Correctness
    # -------------------------------------------------

    print("\n--- CORRECTNESS ---")

    print(
        "Individual:",
        individual_tokens
    )

    print(
        "Batched:",
        batched_tokens.tolist()
    )

    print(
        "Outputs identical:",
        individual_tokens
        == batched_tokens.tolist()
    )

    # -------------------------------------------------
    # Batched cache
    # -------------------------------------------------

    print("\n--- BATCHED CACHE ---")

    print(
        "Layer 0 K:",
        batched_cache.layers[0].keys.shape
    )

    print(
        "Layer 0 V:",
        batched_cache.layers[0].values.shape
    )


if __name__ == "__main__":
    main()