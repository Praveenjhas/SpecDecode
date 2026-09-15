from src.cache.prefix_cache import PrefixCache


def main():

    cache = PrefixCache(
        block_size=4
    )

    cache.put_block(
        [1, 2, 3, 4],
        "KV_BLOCK_0"
    )

    cache.put_block(
        [5, 6, 7, 8],
        "KV_BLOCK_1"
    )

    print("\n--- BLOCK PREFIX CACHE TEST ---")

    length, blocks = cache.get_longest_prefix(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    )

    print("Query:")
    print([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    print("\nReusable prefix length:")
    print(length)

    print("\nNumber of cached blocks:")
    print(len(blocks))

    print("\nCached blocks:")
    print(blocks)

    # Partial first block mismatch
    length, blocks = cache.get_longest_prefix(
        [1, 2, 3, 9, 5, 6, 7, 8]
    )

    print("\n--- MISMATCH TEST ---")

    print("Reusable prefix length:")
    print(length)

    print("Number of cached blocks:")
    print(len(blocks))

    # Second block mismatch
    length, blocks = cache.get_longest_prefix(
        [1, 2, 3, 4, 5, 6, 9, 8]
    )

    print("\n--- SECOND BLOCK MISMATCH ---")

    print("Reusable prefix length:")
    print(length)

    print("Number of cached blocks:")
    print(len(blocks))

    print("\nCache size:")
    print(cache.size())


if __name__ == "__main__":
    main()