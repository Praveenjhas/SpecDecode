from src.cache.prefix_cache import PrefixCache


def main():

    cache = PrefixCache()

    # Fake cache objects for this test.
    # We are testing the lookup logic, not the model.
    cache_a = "KV_FOR_PREFIX_A"
    cache_b = "KV_FOR_PREFIX_AB"

    cache.put(
        [1, 2, 3],
        cache_a
    )

    cache.put(
        [1, 2, 3, 4, 5],
        cache_b
    )

    print("\n--- PREFIX CACHE TEST ---")

    length, value = cache.get_longest_prefix(
        [1, 2, 3, 4, 9]
    )

    print("Query: [1, 2, 3, 4, 9]")
    print("Matched length:", length)
    print("Matched cache:", value)

    length, value = cache.get_longest_prefix(
        [1, 2, 3, 8]
    )

    print("\nQuery: [1, 2, 3, 8]")
    print("Matched length:", length)
    print("Matched cache:", value)

    length, value = cache.get_longest_prefix(
        [9, 9, 9]
    )

    print("\nQuery: [9, 9, 9]")
    print("Matched length:", length)
    print("Matched cache:", value)

    print("\nCache size:", cache.size())


if __name__ == "__main__":
    main()