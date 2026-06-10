# HIP-Graph-Safe Flash-Attention for TurboQuant KV (the hero patch)

This is the core contribution of the repository — submitted upstream as
[TheTom/llama-cpp-turboquant#176](https://github.com/TheTom/llama-cpp-turboquant/pull/176). It makes
quantized (TurboQuant) KV cache coexist with **HIP graphs** on AMD RDNA4, giving fast prefill
*and* a usable decode without crashing.

> **Framing note (2026-06-10):** This patch fixes a **fork-specific regression** in TheTom's
> llama-cpp-turboquant, not an upstream llama.cpp bug. The upstream `launch_fattn` does not
> allocate f16 dequant buffers at runtime (they are pre-computed in the destination tensor's
> extra space), making it graph-capture-safe by design. The batch-aware routing (`Q->ne[1] <= 8`
> → VEC) also already exists upstream (with threshold 2). Our contribution is making TurboQuant
> work on RDNA4 within TheTom's fork — the first known working setup. No upstream llama.cpp PR
> is planned; the target is TheTom's fork only.

Patch file: [`../patches/0001-turbo4-hip-graph-safe-fattn.patch`](../patches/0001-turbo4-hip-graph-safe-fattn.patch)
(against commit `7d9715f`).

---

## The two coupled problems

### Problem A — slow prefill (forced VEC kernel)

The fork forced the **VEC** Flash-Attention kernel for *all* quantized KV types:

```cpp
// fattn.cu, original
if ((ggml_is_quantized(K->type) || ggml_is_quantized(V->type)) && can_use_vector_kernel) {
    return BEST_FATTN_KERNEL_VEC;
}
```

VEC does inline dequant with zero temp-buffer overhead — perfect for decode — but it processes
queries sequentially, which is slow for the large batches used in **prefill** (~188 tok/s).

### Problem B — decode crash under HIP graphs

For the TILE/MMA path, the f16 dequant temp buffers (`K_f16`, `V_f16` in `launch_fattn`) were
allocated with raw `cudaMalloc` / `cudaFree` / `cudaStreamSynchronize`:

```cpp
// fattn-common.cuh, original
struct hip_f16_alloc {
    half * ptr = nullptr;
    cudaStream_t stream;
    void alloc(size_t n) { CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(half))); }
    ~hip_f16_alloc() { /* cudaStreamSynchronize + cudaFree */ }
};
```

These calls are **forbidden during HIP graph capture**. With `GGML_HIP_GRAPHS=ON`, decode
(which is captured into a graph) crashed immediately:

```
FLASH_ATTN_EXT failed: operation not permitted when stream is capturing
```

The raw alloc/free existed for a reason: on gfx1201 there is **no VMM**, and the legacy CUDA
pool retains peak-sized allocations permanently. For a multi-GB f16 dequant buffer that would
negate the quantized-KV VRAM savings → OOM. So we can't simply always use the pool, and we
can't always use raw alloc. The path must depend on whether we are capturing.

---

## The fix — capture-aware allocation + batch-aware kernel choice

### Part 1 — route decode to VEC, prefill to TILE (`fattn.cu`)

Make the force-VEC condition batch-aware:

```cpp
// fattn.cu, patched
if ((ggml_is_quantized(K->type) || ggml_is_quantized(V->type))
        && can_use_vector_kernel && Q->ne[1] <= 8) {
    return BEST_FATTN_KERNEL_VEC;   // decode: graph-safe, inline dequant, no temp buffer
}
// large prefill batches fall through to the fast TILE/MMA kernel (~3.4x faster)
```

- **Decode** (batch ≤ 8) → VEC: graph-safe, supports TurboQuant natively, zero temp buffer.
- **Prefill** (large batch) → TILE/MMA: fast, runs eagerly (not captured), so its raw temp
  buffer allocation is legal.

### Part 2 — capture-aware f16 alloc (`fattn-common.cuh`)

Choose pool vs raw alloc based on capture status:

```cpp
// fattn-common.cuh, patched
cudaStreamCaptureStatus fa_capture_status = cudaStreamCaptureStatusNone;
CUDA_CHECK(cudaStreamIsCapturing(main_stream, &fa_capture_status));
const bool fa_use_pool = (fa_capture_status != cudaStreamCaptureStatusNone) || (Q->ne[1] <= 8);

struct hip_f16_alloc {
    half           * ptr      = nullptr;
    ggml_cuda_pool * mem_pool = nullptr;   // non-null => allocate from pool (graph-safe)
    size_t           pool_size = 0;
    cudaStream_t     stream;
    void alloc(size_t n) {
        if (mem_pool) ptr = (half *) mem_pool->alloc(n * sizeof(half), &pool_size);
        else          CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(half)));
    }
    // dtor: pool->free (no CUDA calls, safe during capture) OR raw sync+free when eager
};
hip_f16_alloc K_f16(main_stream, fa_use_pool ? &pool : nullptr);
hip_f16_alloc V_f16(main_stream, fa_use_pool ? &pool : nullptr);
```

- **During capture / decode:** use `mem_pool->alloc/free` — pure bookkeeping, no `cudaMalloc`
  / `cudaFree` / `cudaStreamSynchronize`, so capture is legal.
- **Large prefill (eager):** raw `cudaMalloc` / `cudaFree` — frees the multi-GB buffer
  immediately, no pool retention, no OOM.

This is safe because ggml resets graph warmup whenever tensor sizes change
(`ggml-cuda.cu`), so the pool buffer is always warmed at the right size before capture and the
in-capture `alloc()` just reuses it.

---

## Result

| Build | Prefill (pp2048) | turbo4 KV | HIP graphs | Decode crash |
|-------|------------------|-----------|------------|--------------|
| jagsan-cyber (no turbo) | 641 t/s | ❌ | ✅ | — |
| TheTom no-graphs fallback | 188 t/s | ✅ | ❌ | none (but slow) |
| **TheTom + this patch** | **735 t/s** | ✅ | ✅ | **none** |

Fast prefill, working TurboQuant KV, HIP graphs on — all at once on RDNA4.

---

## Companion patch — Windows HIP build fixes

[`../patches/0002-windows-hip-build-fixes.patch`](../patches/0002-windows-hip-build-fixes.patch)
covers two small Windows/HIP build issues unrelated to performance:

- `ggml-cuda.cu`: drop the peer-to-peer `cudaMemcpy3DPeerAsync` path (not available on Windows
  HIP) and always use the staging-buffer fallback.
- `vendors/hip.h`: add the missing `cudaEventCreate → hipEventCreate` mapping.
