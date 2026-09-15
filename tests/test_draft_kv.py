from src.model.draft_model_setup import load_draft_model
from src.inference.draft_kv_decoder import DraftKVDecoder


def main():

    tokenizer, model = load_draft_model()

    decoder = DraftKVDecoder(
        model,
        tokenizer
    )

    prompt = (
        "Explain how a CPU executes an instruction."
    )

    input_ids = tokenizer(
        prompt,
        return_tensors="pt"
    )["input_ids"].to("cuda")

    print("\n--- DRAFT PREFILL ---")

    next_token, cache = decoder.prefill(
        input_ids
    )

    print(
        "Prompt tokens:",
        input_ids.shape[1]
    )

    print(
        "Layer 0 K:",
        cache.layers[0].keys.shape
    )

    print(
        "Layer 0 V:",
        cache.layers[0].values.shape
    )

    print(
        "First draft token:",
        repr(
            tokenizer.decode(
                [next_token.item()]
            )
        )
    )

    print("\n--- DRAFT KV GENERATION ---")

    proposed_tokens, final_cache = (
        decoder.generate_k(
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
            tokenizer.decode(
                proposed_tokens[0],
                skip_special_tokens=True
            )
        )
    )

    print(
        "Final Layer 0 K:",
        final_cache.layers[0].keys.shape
    )

    print(
        "Final Layer 0 V:",
        final_cache.layers[0].values.shape
    )


if __name__ == "__main__":
    main()