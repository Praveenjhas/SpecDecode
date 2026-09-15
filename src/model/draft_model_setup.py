import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


DRAFT_MODEL_NAME = "Qwen/Qwen2.5-0.5B"


def load_draft_model():
    print(f"Loading draft model: {DRAFT_MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(
        DRAFT_MODEL_NAME
    )

    model = AutoModelForCausalLM.from_pretrained(
        DRAFT_MODEL_NAME,
        dtype=torch.float16
    )

    model = model.to("cuda")
    model.eval()

    return tokenizer, model