import torch
from src.model.model_setup import load_model


def main():
    tokenizer, model = load_model()

    print("\n--- MODEL CONFIG ---")

    print("Hidden size:", model.config.hidden_size)
    print("Number of layers:", model.config.num_hidden_layers)
    print("Attention heads:", model.config.num_attention_heads)
    print("KV heads:", model.config.num_key_value_heads)

    head_dim = (
        model.config.hidden_size
        // model.config.num_attention_heads
    )

    print("Head dimension:", head_dim)

    prompt = "Hello, how are you?"

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    inputs = {
        key: value.to("cuda")
        for key, value in inputs.items()
    }

    print("\nInput IDs shape:")
    print(inputs["input_ids"].shape)

    with torch.no_grad():
        outputs = model(
            **inputs,
            use_cache=True
        )

    cache = outputs.past_key_values

    print("\n--- KV CACHE ---")
    print("Cache type:", type(cache))

    print("Number of cache layers:", len(cache.layers))

    layer = cache.layers[0]

    key = layer.keys
    value = layer.values

    print("\nLayer 0:")
    print("K shape:", key.shape)
    print("V shape:", value.shape)

    print("\n--- CACHE INTERPRETATION ---")

    print("Batch size:", key.shape[0])
    print("KV heads:", key.shape[1])
    print("Sequence length:", key.shape[2])
    print("Head dimension:", key.shape[3])


if __name__ == "__main__":
    main()