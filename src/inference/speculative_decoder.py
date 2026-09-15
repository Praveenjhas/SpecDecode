import torch
from transformers.cache_utils import DynamicCache

from src.inference.draft_kv_decoder import DraftKVDecoder


class SpeculativeDecoder:

    def __init__(
        self,
        target_model,
        draft_model,
        tokenizer
    ):
        self.target_model = target_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer

        self.draft_decoder = DraftKVDecoder(
            draft_model,
            tokenizer
        )

    @torch.no_grad()
    def target_prefill(self, input_ids):
        """
        Process the complete current sequence with the target model.

        Returns:
            first_target_token
            target_cache
        """

        outputs = self.target_model(
            input_ids=input_ids,
            use_cache=True
        )

        first_target_token = torch.argmax(
            outputs.logits[:, -1, :],
            dim=-1
        )

        return (
            first_target_token,
            outputs.past_key_values
        )

    def _copy_cache(self, cache):
        """
        Create an independent DynamicCache copy.
        """

        new_cache = DynamicCache()

        for layer_idx, layer in enumerate(cache.layers):

            new_cache.update(
                layer.keys.clone(),
                layer.values.clone(),
                layer_idx=layer_idx
            )

        return new_cache

    def _truncate_cache(self, cache, length):
        """
        Keep only the first `length` cached tokens.
        """

        truncated_cache = DynamicCache()

        for layer_idx, layer in enumerate(cache.layers):

            keys = layer.keys[:, :, :length, :].clone()
            values = layer.values[:, :, :length, :].clone()

            truncated_cache.update(
                keys,
                values,
                layer_idx=layer_idx
            )

        return truncated_cache

    @torch.no_grad()
    def verify_with_cache(
        self,
        proposed_tokens,
        target_cache,
        first_target_token
    ):
        """
        Verify the proposed tokens against the target model.

        The first proposed token is compared with the target's
        prefill prediction.
        """

        verification_cache = self._copy_cache(
            target_cache
        )

        outputs = self.target_model(
            input_ids=proposed_tokens,
            past_key_values=verification_cache,
            use_cache=True
        )

        target_logits = outputs.logits

        proposed_length = proposed_tokens.shape[1]

        target_predictions = torch.empty(
            (1, proposed_length),
            dtype=torch.long,
            device=proposed_tokens.device
        )

        # Target prediction for proposed token 0.
        target_predictions[:, 0] = first_target_token

        # Target prediction for proposed tokens 1..K-1.
        if proposed_length > 1:

            target_predictions[:, 1:] = torch.argmax(
                target_logits[:, :-1, :],
                dim=-1
            )

        # Target prediction after the final proposed token.
        next_target_token = torch.argmax(
            target_logits[:, -1, :],
            dim=-1
        )

        return (
            target_predictions,
            next_target_token,
            outputs.past_key_values
        )

    @torch.no_grad()
    def speculative_step(
        self,
        input_ids,
        target_cache,
        first_target_token,
        k=4
    ):
        """
        Execute one speculative iteration.

        Returns:
            committed_tokens
            new_target_cache
            stats
        """

        original_length = (
            target_cache.layers[0].keys.shape[2]
        )

        # -------------------------------------------------
        # 1. Draft proposes K tokens
        # -------------------------------------------------

        proposed_tokens, _ = (
            self.draft_decoder.generate_k(
                input_ids,
                k
            )
        )

        proposed_count = proposed_tokens.shape[1]

        # -------------------------------------------------
        # 2. Target verifies
        # -------------------------------------------------

        (
            target_predictions,
            next_target_token,
            verified_cache
        ) = self.verify_with_cache(
            proposed_tokens,
            target_cache,
            first_target_token
        )

        # -------------------------------------------------
        # 3. Accept until first mismatch
        # -------------------------------------------------

        accepted = []
        mismatch_index = None

        for i in range(proposed_count):

            draft_token = (
                proposed_tokens[0, i].item()
            )

            target_token = (
                target_predictions[0, i].item()
            )

            if draft_token == target_token:

                accepted.append(draft_token)

            else:

                mismatch_index = i

                # Reject draft token and use target token.
                accepted.append(target_token)

                break

        committed_count = len(accepted)

        final_length = (
            original_length + committed_count
        )

        # -------------------------------------------------
        # 4. Roll back target cache
        # -------------------------------------------------

        new_target_cache = self._truncate_cache(
            verified_cache,
            final_length
        )

        committed_tokens = torch.tensor(
            [accepted],
            dtype=torch.long,
            device=input_ids.device
        )

        accepted_draft_count = (
            mismatch_index
            if mismatch_index is not None
            else proposed_count
        )

        stats = {
            "proposed_tokens": proposed_count,
            "accepted_draft_tokens":
                accepted_draft_count,
            "output_tokens":
                committed_count,
            "mismatch_index":
                mismatch_index,
            "next_target_token":
                next_target_token.item(),
        }

        return (
            committed_tokens,
            new_target_cache,
            stats
        )

    @torch.no_grad()
    def generate(
        self,
        prompt,
        max_new_tokens=50,
        draft_k=4
    ):
        """
        Correctness-first speculative generation.

        NOTE:
        The current prototype re-prefills the target between
        speculative iterations so that the target's next-token
        prediction is always explicit and correct.

        This is intentionally not the final performance version.
        """

        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt"
        )["input_ids"].to("cuda")

        generated_tokens = []

        total_proposed = 0
        total_accepted_draft = 0
        target_iterations = 0

        while (
            len(generated_tokens)
            < max_new_tokens
        ):

            # ---------------------------------------------
            # Target prefill on current sequence
            # ---------------------------------------------

            first_target_token, target_cache = (
                self.target_prefill(
                    input_ids
                )
            )

            # ---------------------------------------------
            # Speculative iteration
            # ---------------------------------------------

            (
                committed_tokens,
                _,
                stats
            ) = self.speculative_step(
                input_ids=input_ids,
                target_cache=target_cache,
                first_target_token=first_target_token,
                k=draft_k
            )

            target_iterations += 1

            total_proposed += (
                stats["proposed_tokens"]
            )

            total_accepted_draft += (
                stats["accepted_draft_tokens"]
            )

            # Don't exceed max_new_tokens.
            remaining = (
                max_new_tokens
                - len(generated_tokens)
            )

            committed_tokens = committed_tokens[
                :, :remaining
            ]

            if committed_tokens.shape[1] == 0:
                break

            new_token_list = (
                committed_tokens[0].tolist()
            )

            generated_tokens.extend(
                new_token_list
            )

            # ---------------------------------------------
            # Append committed tokens to current context
            # ---------------------------------------------

            input_ids = torch.cat(
                [
                    input_ids,
                    committed_tokens
                ],
                dim=1
            )

            # EOS
            if (
                self.tokenizer.eos_token_id
                in new_token_list
            ):
                break

        output_text = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )

        acceptance_rate = (
            total_accepted_draft
            / total_proposed
            if total_proposed > 0
            else 0.0
        )

        stats = {
            "generated_tokens":
                len(generated_tokens),
            "draft_k":
                draft_k,
            "total_proposed_tokens":
                total_proposed,
            "total_accepted_draft_tokens":
                total_accepted_draft,
            "acceptance_rate":
                acceptance_rate,
            "target_iterations":
                target_iterations,
        }

        return output_text, stats