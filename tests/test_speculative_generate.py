import time

from src.model.model_setup import load_model
from src.model.draft_model_setup import load_draft_model
from src.inference.speculative_decoder import SpeculativeDecoder


def main():

    print("\n--- LOAD TARGET ---")

    target_tokenizer, target_model = load_model()

    print("\n--- LOAD DRAFT ---")

    _, draft_model = load_draft_model()

    decoder = SpeculativeDecoder(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=target_tokenizer
    )

    prompt = (
        "Explain how a CPU executes an instruction."
    )

    print("\n--- SPECULATIVE GENERATION ---")

    start = time.perf_counter()

    output_text, stats = decoder.generate(
        prompt=prompt,
        max_new_tokens=20,
        draft_k=4
    )

    if __import__("torch").cuda.is_available():
        __import__("torch").cuda.synchronize()

    end = time.perf_counter()

    print("\nPrompt:")
    print(prompt)

    print("\nGenerated:")
    print(output_text)

    print("\n--- STATS ---")

    print(
        "Generated tokens:",
        stats["generated_tokens"]
    )

    print(
        "Draft K:",
        stats["draft_k"]
    )

    print(
        "Total proposed tokens:",
        stats["total_proposed_tokens"]
    )

    print(
        "Total accepted draft tokens:",
        stats["total_accepted_draft_tokens"]
    )

    print(
        "Acceptance rate:",
        round(
            stats["acceptance_rate"] * 100,
            2
        ),
        "%"
    )

    print(
        "Target iterations:",
        stats["target_iterations"]
    )

    print(
        "Latency:",
        round(end - start, 4),
        "seconds"
    )


if __name__ == "__main__":
    main()