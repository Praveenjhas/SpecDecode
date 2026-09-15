from src.model.model_setup import load_model
from src.inference.kv_decoder import KVDecoder
from src.cache.prefix_cache import PrefixCache


def main():

    tokenizer, model = load_model()

    prefix_cache = PrefixCache()

    decoder = KVDecoder(
        model,
        tokenizer,
        prefix_cache
    )

    prompt_a = "Explain computer networks."
    prompt_b = "Explain computer networks."

    print("\n--- PREFIX INFERENCE TEST ---")

    # -------------------------
    # Request A
    # -------------------------

    input_a, token_a, cache_a = decoder.prefill(
        prompt_a
    )

    tokens_a = input_a[0].tolist()

    print("\nRequest A")
    print("Token count:", len(tokens_a))

    length, cached = prefix_cache.get_longest_prefix(
        tokens_a
    )

    print("Cached prefix length:", length)
    print("Cache found:", cached is not None)

    # -------------------------
    # Request B
    # -------------------------

    input_b, token_b, cache_b = decoder.prefill(
        prompt_b
    )

    tokens_b = input_b[0].tolist()

    print("\nRequest B")
    print("Token count:", len(tokens_b))

    length, cached = prefix_cache.get_longest_prefix(
        tokens_b
    )

    print("Cached prefix length:", length)
    print("Cache found:", cached is not None)

    # -------------------------
    # Compare tokenization
    # -------------------------

    print("\n--- TOKEN COMPARISON ---")

    print("A tokens == B tokens:",
          tokens_a == tokens_b)

    print("A cache is B cache:",
          cache_a is cache_b)


if __name__ == "__main__":
    main()