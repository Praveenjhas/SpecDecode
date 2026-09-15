import torch
from transformers.cache_utils import DynamicCache


class BatchedDecoder:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def _merge_same_length_caches(self, caches):

        if not caches:
            raise ValueError("No caches provided")

        sequence_lengths = [
            cache.layers[0].keys.shape[2]
            for cache in caches
        ]

        if len(set(sequence_lengths)) != 1:
            raise ValueError(
                f"Cannot merge different KV lengths: "
                f"{sequence_lengths}"
            )

        num_layers = len(caches[0].layers)

        merged_cache = DynamicCache(
            config=self.model.config
        )

        for layer_idx in range(num_layers):

            keys = torch.cat(
                [
                    cache.layers[layer_idx].keys
                    for cache in caches
                ],
                dim=0
            )

            values = torch.cat(
                [
                    cache.layers[layer_idx].values
                    for cache in caches
                ],
                dim=0
            )

            merged_cache.update(
                keys,
                values,
                layer_idx=layer_idx
            )

        return merged_cache

    def _split_cache(
        self,
        batched_cache,
        batch_size
    ):

        result = [
            DynamicCache(
                config=self.model.config
            )
            for _ in range(batch_size)
        ]

        num_layers = len(
            batched_cache.layers
        )

        for layer_idx in range(num_layers):

            keys = batched_cache.layers[
                layer_idx
            ].keys

            values = batched_cache.layers[
                layer_idx
            ].values

            for batch_idx in range(batch_size):

                result[batch_idx].update(
                    keys[
                        batch_idx:batch_idx + 1
                    ].clone(),
                    values[
                        batch_idx:batch_idx + 1
                    ].clone(),
                    layer_idx=layer_idx
                )

        return result

    def decode_batch(
        self,
        token_ids,
        caches
    ):

        if len(token_ids) != len(caches):
            raise ValueError(
                "Number of tokens must match "
                "number of caches"
            )

        if not caches:
            raise ValueError("Empty batch")

        sequence_lengths = [
            cache.layers[0].keys.shape[2]
            for cache in caches
        ]

        if len(set(sequence_lengths)) != 1:
            raise ValueError(
                f"Cannot batch different KV lengths: "
                f"{sequence_lengths}"
            )

        sequence_length = sequence_lengths[0]

        input_ids = torch.tensor(
            token_ids,
            dtype=torch.long,
            device="cuda"
        ).unsqueeze(1)

        merged_cache = (
            self._merge_same_length_caches(
                caches
            )
        )

        cache_position = torch.tensor(
            [sequence_length],
            device=input_ids.device
        )

        with torch.no_grad():

            outputs = self.model(
                input_ids=input_ids,
                past_key_values=merged_cache,
                cache_position=cache_position,
                use_cache=True
            )

        next_tokens = torch.argmax(
            outputs.logits[:, -1, :],
            dim=-1
        )

        new_caches = self._split_cache(
            outputs.past_key_values,
            len(caches)
        )

        return (
            next_tokens,
            new_caches
        )