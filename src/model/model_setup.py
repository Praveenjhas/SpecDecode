import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
def load_model():
    print(f"Loading model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16
    )

    model = model.to("cuda")
    model.eval()

    return tokenizer, model