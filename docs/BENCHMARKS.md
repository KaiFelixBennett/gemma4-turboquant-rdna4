# Benchmark Results

All results are for **Gemma-4-31B-it Q4_K_M** (17.05 GiB, 30.7 B params) on an
**AMD Radeon AI PRO R9700** (gfx1201, RDNA4, 32 GB VRAM). Every number below was measured on
this hardware; nothing is extrapolated.

## Hardware & Software

| Component | Specification |
|-----------|--------------|
| GPU | AMD Radeon AI PRO R9700 (gfx1201, RDNA4, 32,624 MiB) |
| CPU | Intel Core Ultra 7 265KF (20 threads) |
| OS | Windows 11 |
| HIP SDK | 7.1, Clang 21 |
| Build | TheTom/llama-cpp-turboquant `7d9715f` + patches in `patches/` (HIP graphs ON) |
| Flash Attention | ON |

---

## 1. Throughput: turbo4 KV + HIP graphs

`llama-bench`, build `bTurboQuant-gfx1201-turbo4-graphs` (HIP_GRAPHS=ON), turbo4/turbo4:

| Test | Prefill | Decode | Note |
|------|---------|--------|------|
| `pp2048 / tg128` | **735.7 t/s** | **22.9 t/s** | turbo4 KV + HIP graphs, no crash, exit 0 |
| interactive CLI | 522.7 t/s | 24.1 t/s | coherent output, no crash |

This beats both the non-turbo jagsan baseline (641 t/s prefill, no turbo4) and the
no-graphs turbo4 fallback (188 t/s prefill) — see [HIP-GRAPH-FIX.md](HIP-GRAPH-FIX.md).

---

## 2. Context sweep (turbo4/turbo4, `-b 16384`)

`llama-bench`, prefill = `-p N -n 0`, decode = `-d N -p 0 -n 128`:

| Context | Prefill (t/s) | Decode (t/s) |
|---------|---------------|--------------|
| 2K | 810.85 ± 2.13 | 21.24 ± 0.28 |
| 4K | 712.56 ± 1.32 | 21.78 ± 1.48 |
| 8K | 652.32 ± 1.32 | 19.99 ± 0.52 |
| 16K | 566.04 ± 0.81 | 19.09 ± 0.90 |
| 32K | 461.42 ± 1.11 | 16.18 ± 0.76 |
| 64K | 335.58 ± 0.75 | 13.82 ± 1.12 |
| 128K | 207.74 ± 1.10 | **⚠️ 1.28 ± 0.00** |

The 128K decode collapse looked like a VRAM wall — but it is a **batch-buffer artifact**, see below.

---

## 3. The batch-buffer fix (the key result)

### VRAM at 128K, idle (load-only measurement)

| Config | Batch | Dedicated VRAM | Spill (shared) |
|--------|-------|----------------|----------------|
| turbo3/turbo3 | 2048 | 20.61 GB | 0.30 GB |
| turbo4/turbo4 | 2048 | 21.56 GB | 0.30 GB |
| turbo4/turbo4 | **16384** | **23.40 GB** | **1.15 GB ⚠️** |

The `-b 16384` Flash-Attention scratch buffer adds ~1.8 GB and spills turbo4 over the 32 GB
edge while idle. The KV difference between turbo3 and turbo4 is only ~1 GB.

### Decode @ 128K — batch is the lever

| Config | Batch | Decode t/s | |
|--------|-------|-----------|---|
| turbo4/turbo4 | 16384 | 1.28 | ❌ spill |
| q8_0/turbo4 | 16384 | 1.16 | ❌ spill |
| **turbo4/turbo4** | **2048** | **6.63** | ✅ **+5.2x — pure batch fix** |
| turbo3/turbo3 | 16384 | 9.75 | faster (smaller KV), see QUALITY.md |

**Dropping `-b 16384` → `-b 2048` alone recovers 5.2x decode at 128K with no quality change.**

---

## 4. Loading the full 256K context

turbo3/turbo3, `-b 2048 -ub 512`, load-only:

| Context | Status | Dedicated VRAM | Spill | Free (of 32 GB) |
|---------|--------|----------------|-------|-----------------|
| 128K | ✅ | 20.61 GB | 0.30 GB | ~11 GB |
| 160K | ✅ | 21.13 GB | 0.36 GB | ~11 GB |
| 192K | ✅ | 21.71 GB | 0.43 GB | ~10 GB |
| 224K | ✅ | 22.29 GB | 0.49 GB | ~10 GB |
| **256K** | ✅ | **22.88 GB** | 0.55 GB | **~9 GB** |

KV grows only ~0.58 GB per 32K — Gemma's SWA caps 5 of every 6 layers at a 1024-token window.

---

## 5. Quality

See [QUALITY.md](QUALITY.md) for the full KL-divergence and needle-in-a-haystack study.
Summary: `q8_0/turbo4` needle 9/9 (recommended), `turbo3/turbo3` needle 9/9 (max context).

---

## Historical baseline (jagsan-cyber fork, broken SWA, q4_0/q4_0)

This is the *starting point* before TheTom's fork and our patches. It used a different fork
with a Gemma-4 SWA-pattern parsing bug (see [SWA-BUG.md](SWA-BUG.md)):

| Context | Prefill t/s | Decode t/s | Total |
|---------|-------------|-----------|-------|
| 16K | 545.5 | 20.4 | 38.1s |
| 32K | 765.2 | 16.4 | 52.8s |
| 64K | 481.0 | 3.7 | 177.0s |
| 128K | 289.8 | 1.4 | 588.5s |

> **Note on attribution.** The 1.4 t/s @128K here was originally blamed entirely on the SWA
> bug. Our later measurements show the 128K decode collapse persists even *with* the SWA fix
> when `-b 16384` is used (1.28 t/s) — so the dominant driver at 128K is the **batch-buffer
> spill**, not the SWA bug. The SWA fix matters for correctness and mid-context efficiency;
> the batch flag matters for the 128K cliff.

---

## Methodology

- **llama-bench**: temperature 0, deterministic; prefill `-p N -n 0`, decode `-d N -p 0 -n 128`.
- **VRAM (load-only)**: start `llama-server`, read `\GPU Process Memory(pid_*)\Dedicated Usage`
  and `Shared Usage` performance counters ~15 s after load. Fast and avoids full decode runs.
- **Needle**: `benchmarks/needle_test.py` against a running server (see QUALITY.md).
- **KLD**: `llama-perplexity --kl-divergence` vs a saved f16 baseline, wikitext-2, `-c 512`.

## Result files

- `results/needle_results.jsonl` — raw needle test records (18, all passed)
- `results/needle-longcontext.md` — needle methodology + results
- `results/api_bench_b8192_broken_swa.json` — historical baseline (broken SWA, q4_0/q4_0)
