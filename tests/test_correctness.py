import torch

from src.model.model_setup import load_model
from src.inference.engine import InferenceEngine
from src.inference.kv_decoder import KVDecoder
from src.inference.batched_decoder import BatchedDecoder
from src.inference.request import RequestState
from src.scheduler.scheduler import Scheduler
from src.cache.prefix_cache import PrefixCache


# ============================================================
# Helpers
# ============================================================

def check(condition, message):
    if not condition:
        raise AssertionError(message)

    print("[PASS]", message)


# ============================================================
# Full-context reference decoder
# ============================================================

@torch.no_grad()
def full_context_greedy_tokens(
    model,
    tokenizer,
    prompt,
    max_new_tokens
):
    """
    Slow reference implementation.

    At every step we run the entire prompt + generated sequence
    through the model and greedily select the next token.

    This is the correctness oracle for KV-cache decoding.
    """

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to(model.device)

    generated = []

    for _ in range(max_new_tokens):

        outputs = model(
            input_ids=input_ids,
            use_cache=False
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
            dtype=input_ids.dtype
        )

        input_ids = torch.cat(
            [
                input_ids,
                next_token_tensor
            ],
            dim=1
        )

        if (
            tokenizer.eos_token_id is not None
            and next_token == tokenizer.eos_token_id
        ):
            break

    return generated


# ============================================================
# TEST 1
# Full-context vs KV-cache
# ============================================================

def test_engine_matches_full_context(
    model,
    tokenizer
):
    print("\n--- TEST 1: KV-CACHE VS FULL-CONTEXT ---")

    prompt = (
        "Explain how a CPU executes an instruction."
    )

    max_new_tokens = 20

    # -------------------------------
    # Reference
    # -------------------------------

    reference_tokens = full_context_greedy_tokens(
        model,
        tokenizer,
        prompt,
        max_new_tokens
    )

    # -------------------------------
    # SpecDecode engine
    # -------------------------------

    engine = InferenceEngine(
        model=model,
        tokenizer=tokenizer
    )

    engine_tokens = engine.generate_tokens(
        prompt,
        max_new_tokens=max_new_tokens
    )

    print("\nReference tokens:")
    print(reference_tokens)

    print("\nEngine tokens:")
    print(engine_tokens)

    check(
        engine_tokens == reference_tokens,
        "KV-cache decoding exactly matches full-context greedy decoding"
    )


# ============================================================
# TEST 2
# Batched vs individual
# ============================================================

def test_batching_correctness(
    model,
    tokenizer
):
    print("\n--- TEST 2: BATCHED VS INDIVIDUAL ---")

    decoder = KVDecoder(
        model,
        tokenizer
    )

    batched_decoder = BatchedDecoder(
        model,
        tokenizer
    )

    prompts = [
        "Explain what a CPU is.",
        "Explain what a CPU is."
    ]

    tokens = []
    caches = []

    for prompt in prompts:

        _, next_token, cache = decoder.prefill(
            prompt
        )

        tokens.append(
            next_token.item()
        )

        caches.append(cache)

    # -------------------------------
    # Verify equal KV lengths
    # -------------------------------

    lengths = [
        cache.layers[0].keys.shape[2]
        for cache in caches
    ]

    check(
        len(set(lengths)) == 1,
        f"Test requests have compatible KV lengths: {lengths}"
    )

    # -------------------------------
    # Individual decoding
    # -------------------------------

    individual = []

    for i in range(len(tokens)):

        next_token, _ = decoder.step(
            torch.tensor(
                tokens[i],
                device="cuda"
            ),
            caches[i]
        )

        individual.append(
            next_token.item()
        )

    # -------------------------------
    # Batched decoding
    # -------------------------------

    batched, _ = batched_decoder.decode_batch(
        tokens,
        caches
    )

    batched = batched.tolist()

    check(
        individual == batched,
        "Batched and individual decoding produce identical tokens"
    )


# ============================================================
# TEST 3
# Prefix cache + LRU
# ============================================================

def test_prefix_cache():

    print("\n--- TEST 3: PREFIX CACHE ---")

    cache = PrefixCache(
        block_size=4,
        max_blocks=2
    )

    cache.put_block(
        [1, 2, 3, 4],
        "A"
    )

    cache.put_block(
        [5, 6, 7, 8],
        "B"
    )

    length, blocks = cache.get_longest_prefix(
        [1, 2, 3, 4, 5, 6, 7, 8, 9]
    )

    check(
        length == 8,
        "Prefix cache finds two consecutive cached blocks"
    )

    check(
        blocks == ["A", "B"],
        "Prefix cache returns correct block state"
    )

    # Touch A so B becomes LRU.

    cache.get_block(
        [1, 2, 3, 4]
    )

    cache.put_block(
        [9, 10, 11, 12],
        "C"
    )

    check(
        cache.get_block(
            [5, 6, 7, 8]
        ) is None,
        "LRU cache evicts the least recently used block"
    )

    check(
        cache.get_block(
            [1, 2, 3, 4]
        ) == "A",
        "Recently used block remains cached"
    )

    check(
        cache.get_block(
            [9, 10, 11, 12]
        ) == "C",
        "New block is cached"
    )


# ============================================================
# TEST 4
# Scheduler lifecycle
# ============================================================

def test_scheduler_lifecycle(
    model,
    tokenizer
):

    print("\n--- TEST 4: SCHEDULER LIFECYCLE ---")

    decoder = KVDecoder(
        model,
        tokenizer
    )

    batched_decoder = BatchedDecoder(
        model,
        tokenizer
    )

    scheduler = Scheduler(
        decoder=decoder,
        batched_decoder=batched_decoder,
        max_running_requests=2
    )

    requests = [

        RequestState(
            "A",
            "Explain what a CPU is.",
            max_new_tokens=3
        ),

        RequestState(
            "B",
            "Explain what a GPU is.",
            max_new_tokens=3
        ),

        RequestState(
            "C",
            "Explain what RAM is.",
            max_new_tokens=3
        )

    ]

    for request in requests:
        scheduler.add_request(request)

    # A and B should be admitted first.

    scheduler.step()

    check(
        len(scheduler.running) <= 2,
        "Scheduler respects max_running_requests"
    )

    # Drain all work.

    while scheduler.has_work():
        scheduler.step()

    check(
        all(
            request.finished
            for request in requests
        ),
        "All requests eventually finish"
    )

    check(
        all(
            request.finish_reason is not None
            for request in requests
        ),
        "Every completed request has a finish reason"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n==============================")
    print("SPECDECODE CORRECTNESS SUITE")
    print("==============================")

    tokenizer, model = load_model()

    # TEST 1
    test_engine_matches_full_context(
        model,
        tokenizer
    )

    # TEST 2
    test_batching_correctness(
        model,
        tokenizer
    )

    # TEST 3
    test_prefix_cache()

    # TEST 4
    test_scheduler_lifecycle(
        model,
        tokenizer
    )

    print("\n==============================")
    print("ALL CORE TESTS PASSED")
    print("==============================")


if __name__ == "__main__":
    main()