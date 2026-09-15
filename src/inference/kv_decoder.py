import torch

from src.cache.prefix_cache import PrefixCache


class KVDecoder:

    def __init__(self, model, tokenizer, prefix_cache=None):
        self.model = model
        self.tokenizer = tokenizer

        self.prefix_cache = (
            prefix_cache
            if prefix_cache is not None
            else PrefixCache(
                block_size=4,
                max_blocks=64
            )
        )

    @torch.no_grad()
    def prefill(self, prompt):

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"].to("cuda")

        token_ids = input_ids[0].tolist()

        prefix_length, block_caches = (
            self.prefix_cache.get_longest_prefix(
                token_ids
            )
        )

        if prefix_length == 0:

            outputs = self.model(
                input_ids=input_ids,
                use_cache=True
            )

            cache = outputs.past_key_values

        else:

            print(
                f"Reusing cached prefix: "
                f"{prefix_length} tokens"
            )

            cache = self._build_cache_from_blocks(
                block_caches
            )

            suffix_ids = input_ids[
                :,
                prefix_length:
            ]

            if suffix_ids.shape[1] > 0:

                suffix_length = suffix_ids.shape[1]

                cache_position = torch.arange(
                    prefix_length,
                    prefix_length + suffix_length,
                    device=input_ids.device
                )

                outputs = self.model(
                    input_ids=suffix_ids,
                    past_key_values=cache,
                    cache_position=cache_position,
                    use_cache=True
                )

                cache = outputs.past_key_values

            else:

                # Entire prompt was cached.
                # We need the next-token logits.
                last_token = input_ids[:, -1:]

                cache_length = (
                    cache.layers[0].keys.shape[2]
                )

                cache_position = torch.tensor(
                    [cache_length],
                    device=input_ids.device
                )

                outputs = self.model(
                    input_ids=last_token,
                    past_key_values=cache,
                    cache_position=cache_position,
                    use_cache=True
                )

                cache = outputs.past_key_values

        next_token = torch.argmax(
            outputs.logits[:, -1, :],
            dim=-1
        )

        blocks = self._extract_blocks(
            token_ids,
            cache
        )

        for block_tokens, block_cache in blocks:

            self.prefix_cache.put_block(
                block_tokens,
                block_cache
            )

        return (
            input_ids,
            next_token,
            cache
        )

    @torch.no_grad()
    def step(self, token, cache):

        cache_length = (
            cache.layers[0].keys.shape[2]
        )

        cache_position = torch.tensor(
            [cache_length],
            device=token.device
        )

        outputs = self.model(
            input_ids=token.view(1, 1),
            past_key_values=cache,
            cache_position=cache_position,
            use_cache=True
        )

        next_token = torch.argmax(
            outputs.logits[:, -1, :],
            dim=-1
        )

        return (
            next_token,
            outputs.past_key_values
        )

    def _extract_blocks(
        self,
        token_ids,
        cache
    ):
        block_size = self.prefix_cache.block_size

        blocks = []

        complete_length = (
            len(token_ids) // block_size
        ) * block_size

        for start in range(
            0,
            complete_length,
            block_size
        ):

            end = start + block_size

            block_tokens = token_ids[
                start:end
            ]

            block_cache = []

            for layer in cache.layers:

                keys = layer.keys[
                    :, :, start:end, :
                ].detach().clone()

                values = layer.values[
                    :, :, start:end, :
                ].detach().clone()

                block_cache.append(
                    (keys, values)
                )

            blocks.append(
                (
                    block_tokens,
                    block_cache
                )
            )

        return blocks

    def _build_cache_from_blocks(
        self,
        block_caches
    ):
        from transformers.cache_utils import DynamicCache

        cache = DynamicCache(
            config=self.model.config
        )

        if not block_caches:
            return cache

        num_layers = len(
            block_caches[0]
        )

        for layer_idx in range(num_layers):

            keys = torch.cat(
                [
                    block[layer_idx][0]
                    for block in block_caches
                ],
                dim=2
            )

            values = torch.cat(
                [
                    block[layer_idx][1]
                    for block in block_caches
                ],
                dim=2
            )

            cache.update(
                keys,
                values,
                layer_idx=layer_idx
            )

        return cache