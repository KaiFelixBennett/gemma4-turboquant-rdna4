# The SWA Bug: Why Gemma-4 Decode Speed Collapses at 64K+ Context

## Summary

Gemma-4 uses **hybrid Sliding Window Attention (SWA)**: 5 of every 6 layers use a sliding window (size=1024), and every 6th layer uses full global attention. This means only ~17% of layers need O(n²) attention, while 83% use efficient O(n·w) local attention.

**The bug**: On Windows builds compiled with MSVC or Clang, `std::transform((const bool*)data, ...)` in `llama-model-loader.cpp` misreads the SWA pattern boolean array from the GGUF file. Instead of correctly parsing which layers are SWA vs global, **all 60 layers are treated as global**.

**The impact**: this bug forces every layer into O(n²) global attention, inflating compute
and KV size ~6x and hurting mid-context efficiency. It is fixed in TheTom's fork.

> **Attribution caveat.** Our original baseline measured 1.4 t/s @128K and blamed it entirely
> on this SWA bug. Later measurements show the 128K decode collapse persists even *with* the
> SWA fix when a large `-b 16384` batch is used (1.28 t/s) — so the dominant driver of the
> 128K cliff is the **batch-buffer spill** (see BENCHMARKS.md), while the SWA fix mainly
> improves correctness and mid-context (16K–64K) efficiency. Both fixes matter; don't conflate
> them.

## Technical Details

### Gemma-4 Hybrid Attention Architecture

Gemma-4-31B has 60 layers with this pattern (repeating every 6 layers):
- Layers 0, 1, 2, 3, 4: **Sliding Window Attention** (window=1024)
- Layer 5: **Global Attention** (full context)

This means:
- 50 layers use O(n·1024) = O(n) attention
- 10 layers use O(n²) attention
- Total: ~O(n·1024·50 + n²·10) ≈ O(n·51K + 10n²)

With the bug (all layers global):
- 60 layers use O(n²) attention
- Total: O(60n²) — **6x more compute** than correct

### The Bug Location

In `llama-model-loader.cpp`, the GGUF boolean array for `attention.sliding_window_pattern` is read using:

```cpp
// BUGGY CODE (MSVC/Clang on Windows):
std::transform(
    (const bool*) data,           // ← Casts uint8_t* to bool*
    (const bool*) data + n_layer, // ← Pointer arithmetic wrong for bool
    swa_layers.begin(),
    [](bool b) { return b; }      // ← Reads garbage on Windows
);
```

The problem: `sizeof(bool)` is implementation-defined. MSVC uses 1 byte per bool, but the GGUF format stores booleans as individual bytes (0 or 1). The `(const bool*)` cast causes the compiler to read the wrong number of bytes, producing garbage SWA patterns.

### The Fix

```cpp
// FIXED CODE:
for (uint32_t i = 0; i < n_layer; ++i) {
    swa_layers[i] = static_cast<const uint8_t*>(data)[i] != 0;
}
```

This reads each byte individually as `uint8_t` and converts to bool explicitly, avoiding the `sizeof(bool)` ambiguity.

### Evidence

All R9700 numbers below were measured on this hardware (full data in
[BENCHMARKS.md](BENCHMARKS.md)); the RTX 5090 figure is a third-party report, not ours.
Nothing here is extrapolated.

| Metric | Broken-SWA baseline | Our build (SWA fix) | Note |
|--------|---------------------|---------------------|------|
| Decode @ 128K (R9700) | 1.4 t/s (q4_0/q4_0, `-b 8192`) | 9.38 ± 0.93 t/s (turbo3, `-b 2048`) | ~6.7×, **but the 128K cliff is driven mainly by the batch-buffer spill, not SWA** (see caveat above) |
| KV cache | all 60 layers treated as global (≈6× over) | 50 of 60 layers capped at the 1024 window | restores Gemma's intended hybrid SWA — the fix's real win is correctness + mid-context efficiency |
| Decode @ 128K (RTX 5090) | — | 61.5 t/s | Reddit report, different hardware — **not measured here**, cross-hardware reference only |

### How to Verify the Bug

Run llama-server with Gemma-4 and check the SWA pattern:

```powershell
# In the server output, look for:
# "sliding_window_pattern = [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, ...]"
# (0 = global, 1 = SWA for Gemma-4)
#
# If ALL values are 0 (global), the bug is present.
# Correct pattern for Gemma-4: every 6th layer is 0 (global), rest are 1 (SWA)
```

## References

- [TheTom/turboquant_plus — cross-model validation](https://github.com/TheTom/turboquant_plus/blob/main/docs/cross-model-validation.md) — where the ISWA (hybrid-SWA) bug was discovered (via cross-model testing — "would never have been found on Qwen alone")
- [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) — the fork that carries the fix
- [llama.cpp issue #21394](https://github.com/ggml-org/llama.cpp/issues/21394) — "Eval bug: Gemma4 attn_rot_k and v = 0" — a *related* Gemma-4 eval issue, not necessarily the same root cause as this SWA-pattern misparse
- [Reddit: Gemma 4 31B at 256K on RTX 5090](https://www.reddit.com/r/LocalLLaMA/comments/1sbdihw/) — cross-hardware reference (256K on a 5090)