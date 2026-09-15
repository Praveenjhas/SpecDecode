import torch

from src.model.model_setup import load_model


def main():

    tokenizer, model = load_model()

    prompt = "Explain what a CPU is."

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to("cuda")

    print("\n--- DEBUG KV CORRECTNESS ---")

    print("Prompt IDs:")
    print(input_ids[0].tolist())

    # -------------------------------------------------
    # Manual prefill
    # -------------------------------------------------

    with torch.no_grad():

        manual = model(
            input_ids=input_ids,
            use_cache=True
        )

    manual_cache = manual.past_key_values

    manual_next = torch.argmax(
        manual.logits[:, -1, :],
        dim=-1
    )

    generated = [
        manual_next.item()
    ]

    print(
        "\nInitial next token:",
        manual_next.item(),
        repr(
            tokenizer.decode(
                [manual_next.item()]
            )
        )
    )

    # -------------------------------------------------
    # Compare step by step
    # -------------------------------------------------

    for step in range(20):

        current_length = (
            manual_cache.layers[0]
            .keys.shape[2]
        )

        # ---------------------------------------------
        # A. Full-context reference
        # ---------------------------------------------

        full_ids = torch.cat(
            [
                input_ids,
                torch.tensor(
                    [generated],
                    dtype=torch.long,
                    device="cuda"
                )
            ],
            dim=1
        )

        with torch.no_grad():

            full_outputs = model(
                input_ids=full_ids,
                use_cache=True
            )

        full_next = torch.argmax(
            full_outputs.logits[:, -1, :],
            dim=-1
        )

        # ---------------------------------------------
        # B. Cached/manual decode
        # ---------------------------------------------

        current_token = torch.tensor(
            [[generated[-1]]],
            dtype=torch.long,
            device="cuda"
        )

        cache_position = torch.tensor(
            [current_length],
            dtype=torch.long,
            device="cuda"
        )

        position_ids = cache_position.unsqueeze(0)

        with torch.no_grad():

            cached_outputs = model(
                input_ids=current_token,
                past_key_values=manual_cache,
                cache_position=cache_position,
                position_ids=position_ids,
                use_cache=True
            )

        cached_next = torch.argmax(
            cached_outputs.logits[:, -1, :],
            dim=-1
        )

        # ---------------------------------------------
        # Compare
        # ---------------------------------------------

        full_id = full_next.item()
        cached_id = cached_next.item()

        print(
            f"\nStep {step + 1}"
        )

        print(
            "Full-context:",
            full_id,
            repr(
                tokenizer.decode([full_id])
            )
        )

        print(
            "KV-cache:",
            cached_id,
            repr(
                tokenizer.decode([cached_id])
            )
        )

        print(
            "Cache length:",
            current_length
        )

        if full_id != cached_id:

            print(
                "\n!!! DIVERGENCE FOUND !!!"
            )

            print(
                "Full sequence length:",
                full_ids.shape[1]
            )

            print(
                "Generated so far:",
                generated
            )

            break

        # Update manual state
        manual_cache = (
            cached_outputs.past_key_values
        )

        generated.append(
            cached_id
        )


if __name__ == "__main__":
    main()