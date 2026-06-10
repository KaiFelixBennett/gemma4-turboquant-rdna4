# Gemma-4-31B Configuration Reference

## Model Specifications

| Parameter | Value |
|-----------|-------|
| Model | Gemma-4-31B-it |
| Architecture | Dense transformer with hybrid SWA |
| Parameters | ~31B |
| Layers | 60 |
| Context Length | 262,144 (256K) |
| SWA Window | 1,024 |
| SWA Pattern | 5 SWA + 1 global per 6 layers |
| Thinking Mode | Native (Gemma-4 is a reasoning model) |
| Quantization | Q4_K_M |

## Recommended Server Configuration

Use [`configs/run_gemma4.ps1`](../configs/run_gemma4.ps1) (self-contained launcher), or pass
these flags to `llama-server`:

| Flag | Value | Why |
|------|-------|-----|
| `--ctx-size` | `131072` (or `262144` for full 256K) | context length |
| `--batch-size` / `--ubatch-size` | `2048` / `512` | **CRITICAL** — a 16384 batch spills VRAM at long context |
| `--flash-attn` | `on` | required for quantized KV |
| `--cache-type-k` / `--cache-type-v` | `q8_0` / `turbo4` | asymmetric: 8-bit keys protect attention routing, turbo4 values save memory |
| `--parallel` | `1` | **CRITICAL** — the `--parallel 4` default swaps KV to CPU RAM at long context |
| `--jinja --reasoning-format auto` | | Gemma-4 is a thinking model; clients must read `reasoning_content` |

Sampling (Google's Gemma-4 recommendations): `--temp 1.0 --top-p 0.95 --top-k 64`.

The non-TurboQuant baseline uses `--cache-type-k q4_0 --cache-type-v q4_0` (the older
jagsan-cyber fork — no turbo, broken SWA).

## Sampling Parameters

Google's official recommendations for Gemma-4 instruct mode:

| Parameter | Value | Notes |
|-----------|-------|-------|
| temperature | 1.0 | Gemma-4 is trained for temp=1.0 |
| top_p | 0.95 | Standard nucleus sampling |
| top_k | 64 | Gemma-4 specific |
| min_p | 0.0 | Not needed with top_k=64 |
| presence_penalty | 0.0 | Community consensus for coding |
| repeat_penalty | 1.0 | No repeat penalty needed |

## KV Cache Comparison

Measured on this hardware (gfx1201). "Same-top-p" is the KL-divergence agreement vs the
f16 baseline at `-c 512`; "Needle" is long-context retrieval at 8K–33K (see docs/QUALITY.md).
VRAM is the measured dedicated GPU memory at 128K idle with `-b 2048`.

| Config (K/V) | Same-top-p @512 | Needle 8K–33K | VRAM @128K | Notes |
|--------------|-----------------|---------------|-----------|-------|
| f16 / f16 | 100% | — | ~29–31 GB (calculated, not load-tested) | baseline reference; no practical headroom on a 32 GB card |
| q8_0 / q8_0 | 87.2% | — | ~21 GB | excellent, larger KV |
| **q8_0 / turbo4** | **76.5%** | **9/9** | 21.6 GB | **recommended default** |
| turbo4 / turbo4 | 74.3% | — | 21.6 GB | good; symmetric works here |
| q8_0 / turbo3 | 47.5% | — | 20.6 GB | drifts @512, fine at long ctx |
| turbo3 / turbo3 | 41.0% | **9/9** | 20.6 GB | max context/speed; lossless retrieval |

**Key insights:**
- On Q4_K_M models, K precision protects attention routing via softmax; V compression is
  nearly free. `q8_0-K + turbo4-V` is the safe high-fidelity default.
- The low `turbo3` same-top-p @512 is a **regime artifact**: at 512 tokens the context is
  smaller than Gemma's 1024 SWA window, so the long-context KV path is never exercised. The
  needle test (9/9) shows `turbo3/turbo3` is lossless for long-context retrieval.
- `turbo3` and symmetric `turbo4` run fine on gfx1201 — **no NaNs** in any KLD or needle run.
  Earlier "turbo3 NaN on AMD" reports did not reproduce here.

## SWA Pattern for Gemma-4

Gemma-4-31B has 60 layers in a repeating hybrid pattern: **5 sliding-window layers
(window = 1024) + 1 global (full-attention) layer per 6**, i.e. 50 SWA + 10 global. Only
1/6 of layers need full O(n²) attention, which is what makes long context feasible — provided
the SWA pattern is parsed correctly (see [SWA-BUG.md](SWA-BUG.md)).
