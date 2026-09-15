import torch


class DraftKVDecoder:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def prefill(self, input_ids):
        """
        Process the initial prompt with the draft model
        and create the draft KV cache.
        """

        outputs = self.model(
            input_ids=input_ids,
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

    @torch.no_grad()
    def step(self, token, cache):
        """
        Process exactly one new token using the
        existing draft KV cache.
        """

        outputs = self.model(
            input_ids=token.view(1, 1),
            past_key_values=cache,
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

    @torch.no_grad()
    def generate_k(self, input_ids, k):
        """
        Generate up to k draft tokens using KV caching.

        Returns:
            proposed_tokens
            final_cache
        """

        next_token, cache = self.prefill(
            input_ids
        )

        proposed = []

        for _ in range(k):

            proposed.append(
                next_token.item()
            )

            if (
                next_token.item()
                == self.tokenizer.eos_token_id
            ):
                break

            next_token, cache = self.step(
                next_token,
                cache
            )

        proposed_tokens = torch.tensor(
            [proposed],
            dtype=torch.long,
            device=input_ids.device
        )

        return proposed_tokens, cache