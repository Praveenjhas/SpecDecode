from src.cache.prefix_cache import PrefixCache


def main():

    cache = PrefixCache(
        block_size=4,
        max_blocks=2
    )

    print("\n--- LRU PREFIX CACHE TEST ---")

    # Insert two blocks
    cache.put_block(
        [1, 2, 3, 4],
        "KV_BLOCK_A"
    )

    cache.put_block(
        [5, 6, 7, 8],
        "KV_BLOCK_B"
    )

    print("\nAfter inserting A and B:")
    print(cache.stats())

    # Access A.
    # A becomes most recently used.
    cache.get_block(
        [1, 2, 3, 4]
    )

    print("\nAfter accessing A:")
    print(cache.stats())

    # Insert C.
    # B should be evicted because B is now LRU.
    cache.put_block(
        [9, 10, 11, 12],
        "KV_BLOCK_C"
    )

    print("\nAfter inserting C:")
    print(cache.stats())

    print("\nLookup A:")
    print(
        cache.get_block(
            [1, 2, 3, 4]
        )
    )

    print("\nLookup B:")
    print(
        cache.get_block(
            [5, 6, 7, 8]
        )
    )

    print("\nLookup C:")
    print(
        cache.get_block(
            [9, 10, 11, 12]
        )
    )

    print("\nFinal stats:")
    print(cache.stats())


if __name__ == "__main__":
    main()