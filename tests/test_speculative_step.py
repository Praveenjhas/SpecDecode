from src.model.model_setup import load_model
from src.model.draft_model_setup import load_draft_model
from src.inference.speculative_decoder import SpeculativeDecoder


def main():

    target_tokenizer, target_model = load_model()

    _, draft_model = load_draft_model()

    decoder = SpeculativeDecoder(
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

    first_target_token, target_cache = (
        decoder.target_prefill(
            input_ids
        )
    )

    print("\n--- BEFORE ---")

    print(
        "Prompt length:",
        input_ids.shape[1]
    )

    print(
        "First target token:",
        repr(
            target_tokenizer.decode(
                [first_target_token.item()]
            )
        )
    )

    print(
        "Target KV length:",
        target_cache.layers[0].keys.shape[2]
    )

    # -------------------------------------------------
    # Speculative step
    # -------------------------------------------------

    (
        new_tokens,
        new_cache,
        stats
    ) = decoder.speculative_step(
        input_ids=input_ids,
        target_cache=target_cache,
        first_target_token=first_target_token,
        k=4
    )

    print("\n--- SPECULATIVE STEP ---")

    print(
        "Output IDs:",
        new_tokens[0].tolist()
    )

    print(
        "Output text:",
        repr(
            target_tokenizer.decode(
                new_tokens[0],
                skip_special_tokens=True
            )
        )
    )

    print(
        "Proposed:",
        stats["proposed_tokens"]
    )

    print(
        "Accepted draft tokens:",
        stats["accepted_draft_tokens"]
    )

    print(
        "Output tokens:",
        stats["output_tokens"]
    )

    print(
        "Mismatch index:",
        stats["mismatch_index"]
    )

    print(
        "Next target token:",
        stats["next_target_token"]
    )

    print("\n--- AFTER ---")

    print(
        "New target KV length:",
        new_cache.layers[0].keys.shape[2]
    )


if __name__ == "__main__":
    main()