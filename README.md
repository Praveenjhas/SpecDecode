# SpecDecode

A mini LLM inference engine built with PyTorch, implementing KV-cache decoding, dynamic batching, prefix caching, request scheduling, and speculative decoding.

## Features

- Manual **KV-cache decoding**
- **Length-compatible dynamic batching**
- Block-based **prefix KV caching with LRU eviction**
- Capacity-aware **request scheduling**
- **Speculative decoding** with draft-model verification
- FastAPI inference server
- Concurrency and performance benchmarking
- Correctness tests for KV cache, batching, caching, and scheduling

## Architecture

```text
Client
  │
  ▼
FastAPI Server
  │
  ▼
Request Manager / Scheduler
  │
  ├── Prefix KV Cache (LRU)
  │
  └── Inference Engine
        ├── KV Decoder
        ├── Batched Decoder
        └── Speculative Decoder
              │
              ▼
        Qwen2.5-0.5B-Instruct
```

## Performance

Tested on an NVIDIA RTX 3050 Laptop GPU (4 GB VRAM).

| Benchmark                         | Result          |
| --------------------------------- | --------------- |
| Dynamic batching speedup          | **2.99×**       |
| Individual decode latency         | 192.79 ms       |
| Batched decode latency            | **64.53 ms**    |
| Peak aggregate serving throughput | **41.35 tok/s** |
| Peak throughput concurrency       | **4**           |
| p95 latency at concurrency 4      | **2.57 s**      |
| Speculative draft acceptance      | **43.75%**      |

Batching benchmark uses 4 requests with compatible KV-cache lengths.

## Correctness

The core correctness suite verifies:

- Full-context greedy decoding == KV-cache decoding
- Batched decoding == individual decoding
- Prefix cache and LRU eviction
- Scheduler capacity and request lifecycle

All core tests pass.

## Run

<pre class="overflow-visible! px-0!" data-start="1661" data-end="1726"><div class="relative w-full mt-4 mb-1"><div class=""><div class="contents"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="relative h-full w-full border-radius-3xl bg-(--code-block-surface) corner-superellipse/1.1 overflow-clip rounded-3xl [--code-block-surface:var(--bg-elevated-secondary)] dark:[--code-block-surface:var(--composer-surface-primary)] lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class=""><div class="relative"><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼs ͼ16"><div class="cm-scroller"><pre class="cm-content q9tKkq_readonly m-0"><code><span>pip install </span><span class="ͼ12">-r</span><span> requirements.txt
python main_server.py</span></code></pre></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></div></div></div></pre>

Run correctness tests:

<pre class="overflow-visible! px-0!" data-start="1752" data-end="1796"><div class="relative w-full mt-4 mb-1"><div class=""><div class="contents"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="relative h-full w-full border-radius-3xl bg-(--code-block-surface) corner-superellipse/1.1 overflow-clip rounded-3xl [--code-block-surface:var(--bg-elevated-secondary)] dark:[--code-block-surface:var(--composer-surface-primary)] lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class=""><div class="relative"><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼs ͼ16"><div class="cm-scroller"><pre class="cm-content q9tKkq_readonly m-0"><code><span>python </span><span class="ͼ12">-m</span><span> tests.test_correctness</span></code></pre></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></div></div></div></pre>

Run benchmarks:

<pre class="overflow-visible! px-0!" data-start="1815" data-end="1894"><div class="relative w-full mt-4 mb-1"><div class=""><div class="contents"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="relative h-full w-full border-radius-3xl bg-(--code-block-surface) corner-superellipse/1.1 overflow-clip rounded-3xl [--code-block-surface:var(--bg-elevated-secondary)] dark:[--code-block-surface:var(--composer-surface-primary)] lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class=""><div class="relative"><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼs ͼ16"><div class="cm-scroller"><pre class="cm-content q9tKkq_readonly m-0"><code><span>python </span><span class="ͼ12">-m</span><span> tests.benchmark_batching
python </span><span class="ͼ12">-m</span><span> tests.benchmark_server</span></code></pre></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></div></div></div></pre>

## Tech Stack

**Python · PyTorch · Transformers · CUDA · FastAPI · Uvicorn · Threading**
