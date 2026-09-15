import time

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.model.model_setup import load_model
from src.inference.engine import InferenceEngine
from src.server.request_manager import RequestManager


app = FastAPI(
    title="SpecDecode",
    version="1.0.0",
    description=(
        "LLM inference engine with KV caching, "
        "prefix caching, scheduling and batching."
    )
)


# ---------------------------------------------------------
# Load model once
# ---------------------------------------------------------

tokenizer, model = load_model()

engine = InferenceEngine(
    model=model,
    tokenizer=tokenizer,
    max_running_requests=4,
    prefix_block_size=4,
    prefix_max_blocks=64
)

request_manager = RequestManager(engine)


# ---------------------------------------------------------
# Schemas
# ---------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str = Field(
        min_length=1
    )

    max_new_tokens: int = Field(
        default=50,
        ge=1,
        le=512
    )


class GenerateResponse(BaseModel):
    request_id: str
    text: str
    latency_seconds: float
    generated_tokens: int
    finish_reason: str | None


# ---------------------------------------------------------
# Health
# ---------------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "ok",
        "model_loaded": model is not None,
        "cuda_available": torch.cuda.is_available(),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        )
    }


# ---------------------------------------------------------
# Synchronous endpoint
# ---------------------------------------------------------

@app.post(
    "/generate",
    response_model=GenerateResponse
)
def generate(request: GenerateRequest):

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()

    text = engine.generate(
        prompt=request.prompt,
        max_new_tokens=request.max_new_tokens
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end = time.perf_counter()

    generated_tokens = len(
        tokenizer.encode(
            text,
            add_special_tokens=False
        )
    )

    return GenerateResponse(
        request_id="single-request",
        text=text,
        latency_seconds=end - start,
        generated_tokens=generated_tokens,
        finish_reason="max_new_tokens"
    )


# ---------------------------------------------------------
# Batched / scheduled endpoint
# ---------------------------------------------------------

@app.post(
    "/generate/batched",
    response_model=GenerateResponse
)
def generate_batched(request: GenerateRequest):

    start = time.perf_counter()

    state = request_manager.submit(
        prompt=request.prompt,
        max_new_tokens=request.max_new_tokens
    )

    # Wait until scheduler finishes this request.
    state.done_event.wait()

    end = time.perf_counter()

    text = tokenizer.decode(
        state.generated_tokens,
        skip_special_tokens=True
    )

    return GenerateResponse(
        request_id=state.request_id,
        text=text,
        latency_seconds=end - start,
        generated_tokens=len(
            state.generated_tokens
        ),
        finish_reason=state.finish_reason
    )


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

@app.get("/metrics")
def metrics():

    return {
        "prefix_cache": engine.cache_stats(),
        "scheduler": {
            "waiting": len(
                engine.scheduler.waiting
            ),
            "running": len(
                engine.scheduler.running
            )
        },
        "cuda": {
            "available": torch.cuda.is_available(),
            "memory_allocated_gb": (
                torch.cuda.memory_allocated() / 1024**3
                if torch.cuda.is_available()
                else 0.0
            ),
            "memory_reserved_gb": (
                torch.cuda.memory_reserved() / 1024**3
                if torch.cuda.is_available()
                else 0.0
            )
        }
    }