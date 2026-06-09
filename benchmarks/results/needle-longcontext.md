# Needle-in-a-Haystack: Long-Context Retrieval Quality

**Date:** 2026-06-09
**Build:** `7d9715f` (TheTom llama-cpp-turboquant), self-patched for gfx1201 (RDNA4)
**Model:** `gemma-4-31B-it-Q4_K_M.gguf` (17.05 GiB)
**Device:** AMD Radeon AI PRO R9700, 32 GiB VRAM
**Harness:** `benchmarks/needle_test.py` (stdlib only, OpenAI-compatible client)
**Server:** `-ngl 99 -fa on -c 40960 -b 2048 -ub 512`

---

## Why this test (and not KL-divergence)

A short-context KL-divergence test (`-c 512`) ranked `turbo3` as poor
(Same-top-p 41%, see `20260609-kldivergence-quality.md`). That result is
**misleading for KV-cache quantization** for two reasons:

1. **Regime mismatch.** Gemma-4 uses hybrid SWA: 5 of every 6 layers cap
   attention at a 1024-token window. At `-c 512` the context is *smaller than
   the window*, so every layer attends densely — we never exercise the
   long-context regime that KV quantization targets.
2. **Metric mismatch.** Per-token logit drift (KLD) is far stricter than the
   downstream task that actually matters. Google Research evaluated TurboQuant
   on **LongBench / Needle-in-a-Haystack / RULER** (retrieval accuracy), not
   per-token KLD, and reported near-zero loss at 3 bits.

This test reproduces Google's methodology on our hardware: embed a unique fact
("needle": `QUASAR-XXXX`) at depth d inside filler text of N tokens, then ask
the model to retrieve it. A distinct code per cell prevents cache leakage.

---

## Results

Each cell: retrieve the access key from a context of the given token length,
with the needle placed at the given relative depth.

| Context (tokens) | Depth 0.1 | Depth 0.5 | Depth 0.9 |
|------------------|-----------|-----------|-----------|
| **q8_0 / turbo4** | | | |
| 8 054  | PASS | PASS | PASS |
| 16 156 | PASS | PASS | PASS |
| 33 223 | PASS | PASS | PASS |
| **turbo3 / turbo3** | | | |
| 8 054  | PASS | PASS | PASS |
| 16 156 | PASS | PASS | PASS |
| 33 223 | PASS | PASS | PASS |

**q8_0/turbo4: 9/9 (100%) · turbo3/turbo3: 9/9 (100%)**

### End-to-end latency (full prefill + thinking + answer, seconds)

| Context | q8_0/turbo4 | turbo3/turbo3 |
|---------|-------------|---------------|
| 8 054   | 18.6–24.3 | 18.1–26.4 |
| 16 156  | 31.7–44.9 | 31.2–44.8 |
| 33 223  | 71.8–103.6 | 62.0–96.1 |

(Times include the model's internal reasoning tokens; lower at depth 0.9
because the answer-relevant prefix is shorter to re-read.)

---

## Interpretation

- **The KLD@512 verdict on turbo3 does not transfer to long context.** Despite a
  41% same-top-p at 512 tokens, `turbo3/turbo3` retrieves the needle perfectly at
  8k–33k. This confirms both the user's intuition and Google's published claim:
  3-bit KV compression is effectively lossless for long-context *retrieval*.
- **turbo3 is even marginally faster** at 33k (smaller KV cache → less memory
  traffic), with no retrieval penalty on this task.
- **Honest caveat:** single-fact retrieval is a relatively easy probe. Per-token
  logit drift (real, as KLD shows) could still surface on harder long-context
  tasks such as multi-hop reasoning or exact long-form reproduction, where
  turbo4's higher fidelity may matter. This test proves retrieval, not every
  downstream behavior.

## Recommendation

- **Default: `q8_0/turbo4` + `-b 2048 -ub 512`** — highest fidelity, perfect
  retrieval, usable 128k decode (6.63 t/s), loads to 256k.
- **`turbo3/turbo3`** is a validated option when maximum context (256k) and
  decode speed outweigh worst-case per-token fidelity. For retrieval-style
  agentic workloads its quality is indistinguishable from turbo4 here.
