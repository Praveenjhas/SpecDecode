import torch

from src.model.model_setup import load_model
from src.model.draft_model_setup import load_draft_model


def main():

    print("\n--- LOADING TARGET ---")

    target_tokenizer, target_model = load_model()

    print("\nTarget GPU memory:")
    print(
        round(
            torch.cuda.memory_allocated() / 1024**3,
            3
        ),
        "GB"
    )

    print("\n--- LOADING DRAFT ---")

    draft_tokenizer, draft_model = load_draft_model()

    print("\nDraft GPU memory:")
    print(
        round(
            torch.cuda.memory_allocated() / 1024**3,
            3
        ),
        "GB"
    )

    # -------------------------------------------------
    # Tokenizer comparison
    # -------------------------------------------------

    prompt = "Explain how a CPU works."

    target_ids = target_tokenizer(
        prompt,
        return_tensors="pt"
    )["input_ids"]

    draft_ids = draft_tokenizer(
        prompt,
        return_tensors="pt"
    )["input_ids"]

    print("\n--- TOKENIZER CHECK ---")

    print(
        "Target tokens:",
        target_ids[0].tolist()
    )

    print(
        "Draft tokens:",
        draft_ids[0].tolist()
    )

    print(
        "Same tokenization:",
        torch.equal(
            target_ids,
            draft_ids
        )
    )

    print("\n--- MODEL CHECK ---")

    print(
        "Target parameters:",
        sum(
            p.numel()
            for p in target_model.parameters()
        )
    )

    print(
        "Draft parameters:",
        sum(
            p.numel()
            for p in draft_model.parameters()
        )
    )


if __name__ == "__main__":
    main()