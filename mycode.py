from pathlib import Path

path = Path("tests/test_correctness.py")

text = path.read_text(encoding="utf-8")

# Add the reference decoder if it is not already present.
reference_code = '''

def full_context_greedy_tokens(model, tokenizer, prompt, max_new_tokens):
    """
    Slow reference implementation.

    At every step we run the entire prompt + generated sequence
    through the model and greedily select the next token.

    This is our correctness oracle for KV-cache decoding.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)

    generated = []

    with torch.no_grad():
        for _ in range(max_new_tokens):

            outputs = model(
                input_ids=input_ids,
                use_cache=False,
            )

            next_token = int(
                torch.argmax(
                    outputs.logits[:, -1, :],
                    dim=-1
                ).item()
            )

            generated.append(next_token)

            next_token_tensor = torch.tensor(
                [[next_token]],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

            input_ids = torch.cat(
                [input_ids, next_token_tensor],
                dim=1
            )

            if (
                tokenizer.eos_token_id is not None
                and next_token == tokenizer.eos_token_id
            ):
                break

    return generated
'''

# Insert helper after imports.
if "def full_context_greedy_tokens" not in text:
    marker = "\n\n"
    pos = text.find(marker)

    if pos != -1:
        text = text[:pos] + reference_code + text[pos:]
    else:
        text = reference_code + "\n" + text


# Find the existing engine-vs-HF test.
start = text.find("def test_engine_matches_hf")

if start == -1:
    start = text.find("def test_engine")

if start == -1:
    raise RuntimeError(
        "Could not find the existing engine correctness test."
    )

# Find next top-level function, if any.
next_def = text.find("\ndef ", start + 1)

if next_def == -1:
    end = len(text)
else:
    end = next_def


new_test = '''

def test_engine_matches_full_context():
    """
    Primary correctness test.

    SpecDecode KV-cache decoding must produce exactly the same
    greedy token sequence as full-context decoding.
    """

    prompt = "Explain how a CPU executes an instruction."
    max_new_tokens = 20

    reference_tokens = full_context_greedy_tokens(
        model,
        tokenizer,
        prompt,
        max_new_tokens,
    )

    engine_tokens = engine.generate_tokens(
        prompt,
        max_new_tokens=max_new_tokens,
    )

    print("\\nReference tokens:")
    print(reference_tokens)

    print("\\nEngine tokens:")
    print(engine_tokens)

    assert engine_tokens == reference_tokens, (
        "\\nKV-cache decoding diverged from full-context decoding.\\n"
        f"Reference: {reference_tokens}\\n"
        f"Engine:    {engine_tokens}"
    )
'''

text = text[:start] + new_test + text[end:]

path.write_text(text, encoding="utf-8")

print("Updated:", path)
print()
print("Now run:")
print("python -m tests.test_correctness")