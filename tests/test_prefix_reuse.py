import time

from src.model.model_setup import load_model
from src.inference.kv_decoder import KVDecoder
from src.cache.prefix_cache import PrefixCache


def main():

    tokenizer, model = load_model()

    prefix_cache = PrefixCache(
        block_size=4,
        max_blocks=64
    )

    decoder = KVDecoder(
        model,
        tokenizer,
        prefix_cache
    )

    prompt_a = (
        "Explain computer networks. "
        "What is TCP?"
    )

    prompt_b = (
        "Explain computer networks. "
        "What is UDP?"
    )

    # -------------------------------------------------
    # Token comparison
    # -------------------------------------------------

    tokens_a = tokenizer(
        prompt_a,
        return_tensors="pt"
    )["input_ids"][0].tolist()

    tokens_b = tokenizer(
        prompt_b,
        return_tensors="pt"
    )["input_ids"][0].tolist()

    print("\n--- TOKEN COMPARISON ---")

    print("A tokens:")
    print(tokens_a)

    print("\nB tokens:")
    print(tokens_b)

    # -------------------------------------------------
    # Request A
    # -------------------------------------------------

    print("\n--- REQUEST A ---")

    start = time.perf_counter()

    input_a, token_a, cache_a = (
        decoder.prefill(prompt_a)
    )

    end = time.perf_counter()

    print(
        "Prompt tokens:",
        len(tokens_a)
    )

    print(
        "Prefill time:",
        round(end - start, 4),
        "s"
    )

    # -------------------------------------------------
    # Lookup B
    # -------------------------------------------------

    prefix_length, block_caches = (
        prefix_cache.get_longest_prefix(
            tokens_b
        )
    )

    print("\n--- PREFIX LOOKUP FOR B ---")

    print(
        "Reusable prefix length:",
        prefix_length
    )

    print(
        "Reusable blocks:",
        len(block_caches)
    )

    # -------------------------------------------------
    # Request B
    # -------------------------------------------------

    print("\n--- REQUEST B ---")

    start = time.perf_counter()

    input_b, token_b, cache_b = (
        decoder.prefill(prompt_b)
    )

    end = time.perf_counter()

    print(
        "Prompt tokens:",
        len(tokens_b)
    )

    print(
        "Prefill time:",
        round(end - start, 4),
        "s"
    )

    print("\n--- RESULT ---")

    print(
        "Block size:",
        prefix_cache.block_size
    )

    print(
        "Reusable prefix:",
        prefix_length,
        "tokens"
    )

    print(
        "Reusable blocks:",
        len(block_caches)
    )

    print(
        "Cache entries:",
        prefix_cache.size()
    )
    print("\n--- CACHE STATS ---")
    print(prefix_cache.stats())

if __name__ == "__main__":
    main()