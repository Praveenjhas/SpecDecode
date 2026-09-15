from collections import deque, defaultdict


class Scheduler:

    def __init__(
        self,
        decoder,
        batched_decoder,
        max_running_requests=4
    ):
        self.decoder = decoder
        self.batched_decoder = batched_decoder

        self.max_running_requests = max_running_requests

        self.waiting = deque()
        self.running = []

    def add_request(self, request):
        self.waiting.append(request)

    def admit_requests(self):
        """
        Admit waiting requests until capacity is full.
        """

        while (
            self.waiting
            and len(self.running) < self.max_running_requests
        ):
            request = self.waiting.popleft()

            (
                request.input_ids,
                request.current_token,
                request.cache
            ) = self.decoder.prefill(
                request.prompt
            )

            request.add_token(
                request.current_token.item()
            )

            if (
                request.current_token.item()
                == self.decoder.tokenizer.eos_token_id
            ):
                request.mark_finished("eos")
                continue

            self.running.append(request)

    def _group_by_cache_length(self):
        """
        Group currently running requests by KV-cache length.
        """

        groups = defaultdict(list)

        for request in self.running:

            cache_length = (
                request.cache.layers[0]
                .keys.shape[2]
            )

            groups[cache_length].append(request)

        return groups

    def step(self):
        """
        Process all runnable requests.

        Requests with the same KV length are processed
        together in one batched GPU forward pass.
        """

        self.admit_requests()

        if not self.running:
            return []

        groups = self._group_by_cache_length()

        processed_requests = []

        for cache_length, requests in groups.items():

            token_ids = [
                request.current_token.item()
                for request in requests
            ]

            caches = [
                request.cache
                for request in requests
            ]

            # One GPU forward pass for this group.
            next_tokens, new_caches = (
                self.batched_decoder.decode_batch(
                    token_ids,
                    caches
                )
            )

            for i, request in enumerate(requests):

                next_token = next_tokens[i]
                new_cache = new_caches[i]

                request.current_token = next_token
                request.cache = new_cache

                token_id = next_token.item()

                request.add_token(token_id)

                processed_requests.append(request)

        # Remove finished requests after all groups have
        # completed their current batch step.
        still_running = []

        for request in self.running:

            token_id = request.current_token.item()

            if token_id == self.decoder.tokenizer.eos_token_id:

                request.mark_finished("eos")

            elif (
                len(request.generated_tokens)
                >= request.max_new_tokens
            ):

                request.mark_finished("max_new_tokens")

            if not request.finished:
                still_running.append(request)

        self.running = still_running

        return processed_requests

    def has_work(self):
        return bool(
            self.waiting or self.running
        )