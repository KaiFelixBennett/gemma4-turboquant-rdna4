# Benchmark Results

## Test Setup

| Component | Specification |
|-----------|--------------|
| GPU | AMD Radeon AI PRO R9700 (gfx1201, RDNA4, 32GB VRAM) |
| Model | Gemma-4-31B-it Q4_K_M (17.46 GiB) |
| Build | jagsan-cyber/turboquant-rocm-llamacpp (commit 5cf8d492c) |
| KV Cache | q4_0/q4_0 (baseline) |
| Context | 262144 (256K native) |
| Batch | 8192 / ubatch 2048 |
| Flash Attention | ON |
| Sampling | temp=1.0, top_p=0.95, top_k=64 |

## Phase 1: Baseline (Broken SWA, q4_0/q4_0 KV)

API-based benchmark hitting `http://127.0.0.1:8080/v1/chat/completions`. Each level generates 256 tokens.

| Context | Prompt Tokens | Prefill t/s | Decode t/s | Total Time |
|---------|--------------|-------------|-----------|------------|
| 16K | 14,320 | 545.5 | 20.4 | 38.1s |
| 32K | 28,883 | 765.2 | 16.4 | 52.8s |
| 64K | 58,011 | 481.0 | 3.7 | 177.0s |
| 128K | 116,264 | 289.8 | **1.4** | 588.5s |

### Key Observations

1. **Prefill scales reasonably** — 545→765→481→290 t/s. The 32K spike is likely cache warming.
2. **Decode collapses at 64K+** — 20.4→16.4→3.7→1.4 t/s. This is the SWA bug.
3. **At 128K, decode is 1.4 t/s** — generating 256 tokens takes ~588 seconds (nearly 10 minutes).

### Comparison: RTX 5090 with TurboQuant (Fixed SWA)

| Context | RTX 5090 Prefill | RTX 5090 Decode | R9700 Prefill | R9700 Decode |
|---------|-----------------|-----------------|--------------|--------------|
| 128K | 1,429 t/s | **61.5 t/s** | 290 t/s | **1.4 t/s** |
| 256K | 900 t/s | ~61 t/s* | — | — |

*RTX 5090 decode is constant across context lengths (memory-bandwidth bound).

The 44x decode gap at 128K is **not** a hardware difference — it's the SWA bug.

## Phase 2: After Fix (TheTom fork, q8_0-K + turbo4-V)

*Coming soon — build in progress*

### Expected Results

Based on RTX 5090 benchmarks and architectural analysis:

| Context | Expected Prefill | Expected Decode | Expected Time |
|---------|-----------------|-----------------|--------------|
| 16K | ~500-600 t/s | ~18-22 t/s | ~35-40s |
| 32K | ~600-800 t/s | ~15-20 t/s | ~50-60s |
| 64K | ~400-500 t/s | ~12-18 t/s | ~60-80s |
| 128K | ~250-350 t/s | ~8-15 t/s | ~120-200s |
| 256K | ~150-250 t/s | ~5-10 t/s | ~300-500s |

*These are estimates. Actual results depend on SWA fix effectiveness and turbo4-V performance on HIP.*

## Methodology

- **API-based benchmark**: Hits the running llama-server at `127.0.0.1:8080/v1/chat/completions`
- **No separate instance**: Does NOT start its own llama.cpp (avoids VRAM conflicts)
- **PID-lock**: Prevents concurrent benchmark runs
- **Incremental save**: Results saved after each context level
- **Resume support**: Skips already-completed levels on restart
- **Filler text**: Uses repeated "The quick brown fox..." to reach target token count
- **Temperature 0**: Deterministic generation for consistent measurements

## Benchmark Scripts

- `benchmarks/api_benchmark.py` — Gemma-4 benchmark (contexts: 2K, 4K, 8K, 16K, 32K, 64K, 128K)
- `benchmarks/api_benchmark_qwen.py` — Qwen 3.6 27B benchmark (same contexts)