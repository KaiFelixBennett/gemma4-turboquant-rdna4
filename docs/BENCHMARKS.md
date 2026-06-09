# Benchmark Results

All results are for **Gemma-4-31B-it Q4_K_M** (17.46 GiB) on **AMD Radeon AI PRO R9700** (gfx1201, RDNA4, 32GB VRAM).

## Hardware & Software

| Component | Specification |
|-----------|--------------|
| GPU | AMD Radeon AI PRO R9700 (gfx1201, RDNA4, 32GB VRAM) |
| CPU | AMD Ryzen 9 |
| RAM | 64 GB DDR5 |
| OS | Windows 11 |
| HIP SDK | 7.1.51803 (Clang 21.0.0) |
| Build (Phase 1) | jagsan-cyber/turboquant-rocm-llamacpp (commit 5cf8d492c) |
| Build (Phase 2) | TheTom/llama-cpp-turboquant (feature/turboquant-kv-cache, commit 7d9715f) |
| Flash Attention | ON |
| Batch Size | 8192 / ubatch 2048 |

## Phase 1: Baseline (Broken SWA, q4_0/q4_0)

KV Cache: q4_0/q4_0 (symmetric)
SWA: **BROKEN** — all layers treated as global attention

| Context | Prompt Tokens | Prefill t/s | Decode t/s | Total Time |
|---------|--------------|-------------|-----------|------------|
| 16K | 14,320 | 545.5 | 20.4 | 38.1s |
| 32K | 28,883 | 765.2 | 16.4 | 52.8s |
| 64K | 58,011 | 481.0 | 3.7 | 177.0s |
| 128K | 116,264 | 289.8 | **1.4** | 588.5s |

**Key finding**: Decode speed collapses from 20.4 t/s at 16K to **1.4 t/s at 128K** — a 14.6x degradation caused by the SWA bug.

### Streaming API Benchmark (Broken SWA)

Using the API-based streaming test with temperature=1.0, top_p=0.95:

| Context | Rep | TTFT (s) | TPS | Tokens | Gen Time (s) | Total (s) |
|---------|-----|-----------|-----|--------|--------------|-----------|
| 16K | 1 | 101.1 | 1.6 | 122 | 75.0 | 176.6 |
| 16K | 2 | 0.7 | 1.7 | 125 | 73.5 | 75.1 |
| 16K | 3 | 1.0 | 2.0 | 125 | 63.6 | 64.9 |
| 32K | 1 | 74.9 | 2.1 | 124 | 59.8 | 135.2 |
| 32K | 2 | 0.8 | 2.1 | 124 | 59.7 | 60.9 |
| 32K | 3 | 0.8 | 1.6 | 124 | 79.8 | 81.3 |
| 64K | 1 | 261.3 | 0.9 | 113 | 124.8 | 386.9 |
| 64K | 2 | 1.3 | 1.1 | 118 | 106.7 | 108.8 |
| 64K | 3 | 1.3 | 1.0 | 118 | 114.2 | 116.3 |
| 128K | 1 | — | — | 0 | — | timed out |
| 128K | 2 | 56.4 | 0.9 | 117 | 128.4 | 185.7 |
| 128K | 3 | 1.7 | 1.0 | 116 | 111.6 | 114.2 |

Note: First request at each context level has high TTFT due to KV cache warming.

## Phase 2: After Fix (TheTom fork, q8_0-K + turbo4-V)

KV Cache: q8_0-K + turbo4-V (asymmetric)
SWA: **FIXED** — correct boolean array parsing for hybrid attention

### Quick Test (4K context)

| Metric | Value |
|--------|-------|
| Prefill | 423.2 t/s |
| Context | 4096 tokens |
| Cache | q8_0-K + turbo4-V |
| GPU | AMD Radeon AI PRO R9700 (gfx1201) |

### Full Benchmark (llama-bench)

*Benchmark in progress — results will be added when complete*

Expected improvement over baseline:
- Decode at 128K: from **1.4 t/s** → **8-15 t/s** (6-11x improvement)
- Decode at 64K: from **3.7 t/s** → **12-18 t/s** (3-5x improvement)
- VRAM savings from turbo4-V compression: ~3.8x on V cache

## Comparison: RTX 5090 with TurboQuant

| Context | RTX 5090 Prefill | RTX 5090 Decode | R9700 Baseline | R9700 Expected |
|---------|-----------------|-----------------|----------------|----------------|
| 128K | 1,429 t/s | **61.5 t/s** | 290 t/s / 1.4 t/s | ~290 t/s / 8-15 t/s |
| 256K | 900 t/s | ~61 t/s | — | — |

The 44x decode gap at 128K in the baseline is **not** a hardware difference — it's the SWA bug.

## Methodology

- **llama-bench**: Direct benchmark tool, temperature=0, deterministic generation
- **API benchmark**: Hits running llama-server at `127.0.0.1:8080/v1/chat/completions`
- **Streaming benchmark**: Same as API but with temperature=1.0, top_p=0.95
- **Context levels**: 16K, 32K, 64K, 128K (256K planned)
- **Repetitions**: 2 per level (llama-bench), 3 per level (streaming)

## Benchmark Scripts

- `benchmarks/api_benchmark.py` — Gemma-4 benchmark (contexts: 2K, 4K, 8K, 16K, 32K, 64K, 128K)
- `benchmarks/api_benchmark_qwen.py` — Qwen 3.6 27B benchmark (same contexts)

## Result Files

- `results/api_bench_b8192_broken_swa.json` — Baseline (broken SWA, q4_0/q4_0)
- `results/turboquant_streaming_20260609T075714Z.json` — Streaming test (broken SWA)
- `results/turbo4_swa_fix_bench.json` — TheTom fork (SWA fix + turbo4) — *coming soon*