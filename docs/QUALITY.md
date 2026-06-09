# Quality Study: TurboQuant KV Cache on Gemma-4-31B-it

How good is the compressed KV cache, really? We measured it two ways on the actual hardware
(AMD Radeon AI PRO R9700, gfx1201). The headline:

- **`q8_0/turbo4` is the safe high-fidelity default** — needle 9/9, KLD same-top-p 76.5%.
- **`turbo3/turbo3` is lossless for long-context retrieval** — needle 9/9 — despite a poor
  short-context KLD number, because KLD@512 is the wrong regime to judge it.

---

## Method 1 — KL-divergence vs the f16 baseline

Why KLD and not perplexity? On a reasoning/chat model like Gemma-4-it, raw-Wikitext
perplexity is meaningless (PPL ≈ 20,000+, because Wikitext continuation is out of
distribution). The correct test is the **KL-divergence of each quantized KV config against the
f16/f16 baseline of the same model** — it measures directly how much quantization distorts the
output distribution, independent of the model's reasoning character.

**Setup:** wikitext-2-raw test, 20 chunks, `-c 512`, `-fa 1`, `llama-perplexity
--kl-divergence` vs a saved f16 logit baseline. Build `7d9715f`, gfx1201.

| Config (K/V) | Median KLD | Mean KLD | Max KLD | Same-top-p | Tier |
|--------------|-----------|----------|---------|------------|------|
| f16 / f16 | 0 (ref) | 0 | 0 | 100% | baseline |
| q8_0 / q8_0 | 0.0147 | 0.327 | 34.9 | 87.2% | KV quant, no turbo |
| **q8_0 / turbo4** | 0.0996 | 0.792 | 31.2 | **76.5%** | recommended |
| turbo4 / turbo4 | 0.1338 | — | — | 74.3% | symmetric, good |
| q8_0 / turbo3 | 1.8676 | — | — | 47.5% | drifts @512 |
| turbo3 / turbo3 | 3.0985 | — | — | 41.0% | drifts @512 |

### Why `-c 512` is the wrong regime for turbo3

Gemma-4 uses hybrid SWA: 5 of every 6 layers cap attention at a **1024-token window**. At
`-c 512` the context is *smaller than the window*, so every layer attends densely and we never
exercise the long-context KV path that quantization targets. KLD@512 is a strict *stress*
metric — useful to show turbo3 has higher raw per-token drift — but it is **not** a verdict on
long-context behavior.

### Reconciling with Google's "3-bit, near-zero loss"

Google Research reports 3-bit TurboQuant KV with near-zero downstream loss. No contradiction —
we measure something different on four axes:

1. **Metric:** Google measures downstream task accuracy (LongBench / Needle / RULER). We
   measure per-token KLD — a far stricter lens. A model can have 41% same-top-p and still
   answer correctly.
2. **Implementation:** Google's "3-bit zero loss" uses the full algorithm (random rotation +
   PolarQuant + QJL 1-bit error correction). `turbo3` here is the community llama.cpp port —
   not guaranteed bit-identical, especially V-cache.
3. **Context length:** TurboQuant is designed for *long* context. We KLD at `-c 512` (short),
   where error correction has the least to average over.
4. **Model type:** Google tests mostly base models; we test Gemma-4-31B-**it** (reasoning-
   tuned), which has a sharper logit profile that is more sensitive to perturbation.

---

## Method 2 — Needle-in-a-haystack (Google's methodology)

This reproduces the actual long-context task. Embed a unique fact ("needle":
`QUASAR-XXXX`) at relative depth *d* inside *N* tokens of bland filler, then ask the model to
retrieve only the key. A distinct code per cell prevents prompt-cache leakage. The harness
(`benchmarks/needle_test.py`) is stdlib-only and reads both `content` and `reasoning_content`
(Gemma-4 is a thinking model).

**Server:** `-ngl 99 -fa on -c 40960 -b 2048 -ub 512`.

| Context (tokens) | Depth 0.1 | Depth 0.5 | Depth 0.9 |
|------------------|-----------|-----------|-----------|
| **q8_0 / turbo4** | | | |
| 8,054 | PASS | PASS | PASS |
| 16,156 | PASS | PASS | PASS |
| 33,223 | PASS | PASS | PASS |
| **turbo3 / turbo3** | | | |
| 8,054 | PASS | PASS | PASS |
| 16,156 | PASS | PASS | PASS |
| 33,223 | PASS | PASS | PASS |

**q8_0/turbo4: 9/9 · turbo3/turbo3: 9/9**

### End-to-end latency (full prefill + thinking + answer, seconds)

| Context | q8_0/turbo4 | turbo3/turbo3 |
|---------|-------------|---------------|
| 8,054 | 18.6–24.3 | 18.1–26.4 |
| 16,156 | 31.7–44.9 | 31.2–44.8 |
| 33,223 | 71.8–103.6 | 62.0–96.1 |

turbo3 is even marginally faster at 33K (smaller KV → less memory traffic), with no retrieval
penalty.

---

## Interpretation & recommendation

- **The KLD@512 verdict on turbo3 does not transfer to long context.** Despite 41% same-top-p
  at 512 tokens, `turbo3/turbo3` retrieves the needle perfectly at 8K–33K. This confirms both
  the long-context intuition and Google's published claim: 3-bit KV is effectively lossless
  for long-context *retrieval*.
- **Honest caveat:** single-fact retrieval is a relatively easy probe. The per-token logit
  drift KLD reveals is real and could surface on harder tasks (multi-hop reasoning, exact
  long-form reproduction), where turbo4's higher fidelity may matter. This test proves
  retrieval, not every downstream behavior.

**Default:** `q8_0/turbo4` + `-b 2048` — highest fidelity, perfect retrieval, usable 128K
decode (6.63 t/s), loads to 256K. **Max-context option:** `turbo3/turbo3` when 256K and decode
speed outweigh worst-case per-token fidelity.

Raw records: [../benchmarks/results/needle_results.jsonl](../benchmarks/results/needle_results.jsonl).
