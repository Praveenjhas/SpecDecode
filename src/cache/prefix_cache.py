from collections import OrderedDict


class PrefixCache:

    def __init__(self, block_size=4, max_blocks=64):
        self.block_size = block_size
        self.max_blocks = max_blocks

        # key -> block KV cache
        #
        # OrderedDict lets us maintain LRU order.
        # Last item = most recently used.
        self._cache = OrderedDict()

        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def put_block(self, token_ids, block_cache):
        """
        Insert or replace one complete prefix block.
        """

        if len(token_ids) != self.block_size:
            raise ValueError(
                f"Expected block of {self.block_size} tokens, "
                f"got {len(token_ids)}"
            )

        key = tuple(token_ids)

        # Existing entry: update value and mark as recent.
        if key in self._cache:
            self._cache.pop(key)

        self._cache[key] = block_cache

        # Evict least recently used blocks.
        while len(self._cache) > self.max_blocks:
            self._cache.popitem(last=False)
            self.evictions += 1

    def get_block(self, token_ids):
        """
        Return a cached block if present.

        Accessing a block makes it the most recently used.
        """

        key = tuple(token_ids)

        if key not in self._cache:
            self.misses += 1
            return None

        block_cache = self._cache.pop(key)

        # Move to MRU position.
        self._cache[key] = block_cache

        self.hits += 1

        return block_cache

    def get_longest_prefix(self, token_ids):
        """
        Find the longest sequence of consecutive cached blocks
        starting from token 0.

        Returns:
            (reusable_token_count, block_caches)
        """

        token_ids = list(token_ids)

        reusable_length = 0
        block_caches = []

        for start in range(
            0,
            len(token_ids) - self.block_size + 1,
            self.block_size
        ):
            block_tokens = token_ids[
                start:start + self.block_size
            ]

            block_cache = self.get_block(block_tokens)

            if block_cache is None:
                break

            block_caches.append(block_cache)
            reusable_length += self.block_size

        return reusable_length, block_caches

    def size(self):
        return len(self._cache)

    def clear(self):
        self._cache.clear()

        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def stats(self):
        total_lookups = self.hits + self.misses

        hit_rate = (
            self.hits / total_lookups
            if total_lookups > 0
            else 0.0
        )

        return {
            "size": self.size(),
            "capacity": self.max_blocks,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": hit_rate,
        }