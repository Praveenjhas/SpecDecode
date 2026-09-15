from src.inference.kv_decoder import KVDecoder
from src.inference.batched_decoder import BatchedDecoder
from src.inference.request import RequestState
from src.scheduler.scheduler import Scheduler
from src.cache.prefix_cache import PrefixCache


class InferenceEngine:

    def __init__(
        self,
        model,
        tokenizer,
        max_running_requests=4,
        prefix_block_size=4,
        prefix_max_blocks=64
    ):
        self.model = model
        self.tokenizer = tokenizer

        self.prefix_cache = PrefixCache(
            block_size=prefix_block_size,
            max_blocks=prefix_max_blocks
        )

        self.decoder = KVDecoder(
            model=model,
            tokenizer=tokenizer,
            prefix_cache=self.prefix_cache
        )

        self.batched_decoder = BatchedDecoder(
            model=model,
            tokenizer=tokenizer
        )

        self.scheduler = Scheduler(
            decoder=self.decoder,
            batched_decoder=self.batched_decoder,
            max_running_requests=max_running_requests
        )

    def generate_tokens(
        self,
        prompt,
        max_new_tokens=50
    ):
        """
        Generate and return the actual token IDs.

        This is the canonical deterministic generation
        path for correctness testing.
        """

        (
            _,
            current_token,
            cache
        ) = self.decoder.prefill(prompt)

        generated_tokens = []

        for _ in range(max_new_tokens):

            token_id = current_token.item()

            if token_id == self.tokenizer.eos_token_id:
                break

            generated_tokens.append(token_id)

            if len(generated_tokens) >= max_new_tokens:
                break

            current_token, cache = self.decoder.step(
                current_token,
                cache
            )

        return generated_tokens

    def generate(
        self,
        prompt,
        max_new_tokens=50
    ):
        """
        Public text-generation API.
        """

        generated_tokens = self.generate_tokens(
            prompt,
            max_new_tokens
        )

        return self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )

    def submit_request(
        self,
        request_id,
        prompt,
        max_new_tokens=50
    ):
        request = RequestState(
            request_id=request_id,
            prompt=prompt,
            max_new_tokens=max_new_tokens
        )

        self.scheduler.add_request(request)

        return request

    def step(self):
        return self.scheduler.step()

    def has_work(self):
        return self.scheduler.has_work()

    def cache_stats(self):
        return self.prefix_cache.stats()