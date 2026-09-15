from src.model.model_setup import load_model
from src.model.draft_model_setup import load_draft_model
from src.inference.draft_kv_decoder import DraftKVDecoder
from src.inference.speculative_decoder import SpeculativeDecoder


def main():

    print("\n--- LOAD TARGET ---")

    target_tokenizer, target_model = load_model()

    print("\n--- LOAD DRAFT ---")

    draft_tokenizer, draft_model = load_draft_model()

    draft_decoder = DraftKVDecoder(
        draft_model,
        draft_tokenizer
    )

    speculative = SpeculativeDecoder(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=target_tokenizer
    )

    prompt = (
        "Explain how a CPU executes an instruction."
    )

    input_ids = target_tokenizer(
        prompt,
        return_tensors="pt"
    )["input_ids"].to("cuda")

    # -------------------------------------------------
    # Target prefill
    # -------------------------------------------------

    print("\n--- TARGET PREFILL ---")

    target_first_token, target_cache = (
        speculative.target_prefill(
            input_ids
        )
    )

    print(
        "Prompt tokens:",
        input_ids.shape[1]
    )

    print(
        "Target first token:",
        target_first_token.item()
    )

    print(
        "Target first token text:",
        repr(
            target_tokenizer.decode(
                [target_first_token.item()]
            )
        )
    )

    print(
        "Target KV:",
        target_cache.layers[0].keys.shape
    )

    # -------------------------------------------------
    # Draft proposal
    # -------------------------------------------------

    print("\n--- DRAFT ---")

    proposed_tokens, draft_cache = (
        draft_decoder.generate_k(
            input_ids,
            k=4
        )
    )

    print(
        "Proposed IDs:",
        proposed_tokens[0].tolist()
    )

    print(
        "Proposed text:",
        repr(
            target_tokenizer.decode(
                proposed_tokens[0],
                skip_special_tokens=True
            )
        )
    )

    # -------------------------------------------------
    # Target verification
    # -------------------------------------------------

    print("\n--- TARGET VERIFICATION ---")

    (
        target_predictions,
        next_target_token,
        verified_cache
    ) = speculative.verify_with_cache(
        proposed_tokens,
        target_cache,
        target_first_token
    )

    print(
        "Target predictions:",
        target_predictions[0].tolist()
    )

    print(
        "Target prediction text:",
        repr(
            target_tokenizer.decode(
                target_predictions[0],
                skip_special_tokens=True
            )
        )
    )

    print(
        "Next target token:",
        next_target_token.item()
    )

    print(
        "Verified cache:",
        verified_cache.layers[0].keys.shape
    )

    # -------------------------------------------------
    # Compare
    # -------------------------------------------------

    print("\n--- COMPARISON ---")

    for i in range(
        proposed_tokens.shape[1]
    ):

        draft_token = (
            proposed_tokens[0, i].item()
        )

        target_token = (
            target_predictions[0, i].item()
        )

        print(
            f"Position {i}: "
            f"draft={draft_token}, "
            f"target={target_token}, "
            f"match={draft_token == target_token}"
        )


if __name__ == "__main__":
    main()