import torch

from src.model.model_setup import load_model
from src.model.draft_model_setup import load_draft_model
from src.inference.speculative_decoder import SpeculativeDecoder


def main():

    print("\n--- LOADING TARGET ---")

    target_tokenizer, target_model = load_model()

    print("\n--- LOADING DRAFT ---")

    draft_tokenizer, draft_model = load_draft_model()

    # Verify tokenizer compatibility.
    if (
        target_tokenizer.encode(
            "Hello world"
        )
        !=
        draft_tokenizer.encode(
            "Hello world"
        )
    ):
        raise RuntimeError(
            "Target and draft tokenizers are incompatible."
        )

    decoder = SpeculativeDecoder(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=target_tokenizer
    )

    prompt = "Explain how a CPU executes an instruction."

    input_ids = target_tokenizer(
        prompt,
        return_tensors="pt"
    )["input_ids"].to("cuda")

    print("\n--- SPECULATIVE STEP ---")

    print(
        "Prompt tokens:",
        input_ids.shape[1]
    )

    accepted_tokens, stats = (
        decoder.speculative_step(
            input_ids,
            k=4
        )
    )

    print(
        "Accepted token IDs:",
        accepted_tokens[0].tolist()
    )

    print(
        "Accepted text:",
        target_tokenizer.decode(
            accepted_tokens[0],
            skip_special_tokens=True
        )
    )

    print("\n--- STATS ---")

    print(
        "Proposed:",
        stats["proposed_tokens"]
    )

    print(
        "Accepted:",
        stats["accepted_tokens"]
    )

    print(
        "Acceptance rate:",
        round(
            stats["acceptance_rate"] * 100,
            2
        ),
        "%"
    )


if __name__ == "__main__":
    main()