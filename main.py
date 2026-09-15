import time
import torch
from src.model.model_setup import load_model
def main():
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0))

    tokenizer, model = load_model()

    prompt = "Explain how a CPU executes an instruction."

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to("cuda") for key, value in inputs.items()}

    # Make sure previous GPU work is finished
    torch.cuda.synchronize()

    start = time.perf_counter()

    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False
    )

    torch.cuda.synchronize()

    end = time.perf_counter()

    generated_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
    latency = end - start

    text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    print("\n--- OUTPUT ---")
    print(text)

    print("\n--- BASELINE METRICS ---")
    print("Generated tokens:", generated_tokens)
    print("Latency:", round(latency, 4), "seconds")
    print(
        "Tokens/sec:",
        round(generated_tokens / latency, 2)
    )


if __name__ == "__main__":
    main()