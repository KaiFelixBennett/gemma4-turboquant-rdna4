# The SWA Bug: Why Gemma-4 Decode Speed Collapses at 64K+ Context

## Summary

Gemma-4 uses **hybrid Sliding Window Attention (SWA)**: 5 of every 6 layers use a sliding window (size=1024), and every 6th layer uses full global attention. This means only ~17% of layers need O(n²) attention, while 83% use efficient O(n·w) local attention.

**The bug**: On Windows builds compiled with MSVC or Clang, `std::transform((const bool*)data, ...)` in `llama-model-loader.cpp` misreads the SWA pattern boolean array from the GGUF file. Instead of correctly parsing which layers are SWA vs global, **all 60 layers are treated as global**.

**The impact**: At 128K context, decode speed drops from an expected ~40-60 t/s to **1.4 t/s** — a 30-44x degradation.

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

| Metric | Broken SWA | Fixed SWA | Improvement |
|--------|-----------|-----------|-------------|
| Decode @ 128K (RTX 5090) | N/A | 61.5 t/s | — |
| Decode @ 128K (R9700) | 1.4 t/s | ~40-60 t/s (expected) | **30-44x** |
| KV cache size | 6x over-allocated | Correct | 6x reduction |
| VRAM usage | 27.2 GB | ~22 GB (est.) | ~5 GB savings |

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

- [Reddit: Gemma 4 31B at 256K on RTX 5090](https://www.reddit.com/r/LocalLLaMA/comments/1sbdihw/) — Original report of the bug and fix
- [llama.cpp issue #21394](https://github.com/ggml-org/llama.cpp/issues/21394) — "Gemma4 attn_rot_k and v = 0"
- [TheTom/turboquant_plus cross-model validation](https://github.com/TheTom/turboquant_plus/blob/main/docs/cross-model-validation.md) — ISWA bug discovery and fix
- [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) — Fork with the fix