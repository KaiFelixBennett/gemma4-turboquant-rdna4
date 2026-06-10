# PR Notes — TheTom's Fork

> **Status (2026-06-10): submitted as
> [TheTom/llama-cpp-turboquant#176](https://github.com/TheTom/llama-cpp-turboquant/pull/176)** —
> rebased onto upstream tip `73eb521`, containing only the Flash-Attention fix (patch 0001).
> Patch 0002 was dropped from the PR: upstream fixed the Windows build issues independently
> in their #173.

Target: **TheTom/llama-cpp-turboquant** only — both patches are fork-specific. No upstream
ggml-org/llama.cpp PR is planned.

> **Why not upstream?** We checked current ggml-org/llama.cpp master. The decode crash patch
> 0001 fixes **does not exist upstream**: upstream `launch_fattn` doesn't allocate the f16 K/V
> dequant buffers at runtime (they're pre-computed in the dst tensor's extra space → graph-
> capture-safe by design), and the small-batch→VEC routing already exists there. Patch 0002's
> `cudaEventCreate` gap likewise stems from fork-specific TurboQuant code (upstream uses
> `cudaEventCreateWithFlags`). There is no upstream bug to fix.

---

## PR 1 — TheTom/llama-cpp-turboquant

**Target branch:** `feature/turboquant-kv-cache`
**Title:** `HIP: graph-capture-aware Flash-Attention for turbo KV; batch-aware VEC/TILE routing`

### Problem

On AMD HIP with `GGML_HIP_GRAPHS=ON`, running turbo3 or turbo4 KV cache produces an
immediate decode crash:

```
FLASH_ATTN_EXT failed: operation not permitted when stream is capturing
```

Root cause: two coupled issues in `fattn.cu` + `fattn-common.cuh`.

**A — VEC forced for all quantized KV** (`fattn.cu`)

```cpp
// before: applies to ALL batch sizes, including large prefill batches
if ((ggml_is_quantized(K->type) || ggml_is_quantized(V->type)) && can_use_vector_kernel) {
    return BEST_FATTN_KERNEL_VEC;  // ~188 tok/s prefill
}
```

Decode correctly uses VEC (graph-safe, inline dequant). But large prefill batches also
hit VEC, giving ~3.4x slower prefill than the TILE/MMA kernel.

**B — Raw cudaMalloc/cudaFree in launch_fattn** (`fattn-common.cuh`)

The f16 dequant temp buffers (`K_f16`, `V_f16`) were allocated with raw
`cudaMalloc/cudaFree/cudaStreamSynchronize`. These calls are forbidden during HIP graph
capture → crash on first decode step when graphs are active.

### Fix

**A — Batch-aware kernel selection** (`fattn.cu`):
```cpp
// after: VEC only for small batches (decode), TILE for large batches (prefill)
if ((ggml_is_quantized(K->type) || ggml_is_quantized(V->type))
        && can_use_vector_kernel && Q->ne[1] <= 8) {
    return BEST_FATTN_KERNEL_VEC;   // decode: graph-safe, zero temp buffer
}
// large prefill falls through to fast TILE/MMA kernel (~735 tok/s)
```

**B — Capture-aware f16 alloc** (`fattn-common.cuh`):
```cpp
cudaStreamCaptureStatus fa_capture_status;
CUDA_CHECK(cudaStreamIsCapturing(main_stream, &fa_capture_status));
const bool fa_use_pool = (fa_capture_status != cudaStreamCaptureStatusNone) || (Q->ne[1] <= 8);
// ... pool alloc during capture (graph-safe); raw alloc during eager prefill (freed immediately)
```

Safe because ggml resets graph warmup on tensor-size changes, warming the pool buffer at
the right size before capture begins.

### Results (AMD Radeon AI PRO R9700, gfx1201, RDNA4)

| Build / config | Prefill (pp2048) | Decode @ 4K | Crash? |
|----------------|------------------|-------------|--------|
| HIP_GRAPHS=OFF, TheTom (baseline) | 188 tok/s | 22 tok/s | No |
| HIP_GRAPHS=ON, no-patch (turbo4) | — | — | **Yes** |
| **HIP_GRAPHS=ON + this patch (turbo4)** | **735 tok/s** | **22.9 tok/s** | **No** |

**Decode at 128K context (batch-buffer finding):**

| Config | `-b` | Decode @ 128K |
|--------|------|--------------|
| turbo4/turbo4 | 16384 | 1.28 tok/s (VRAM spill) |
| turbo4/turbo4 | **2048** | **6.63 tok/s** (+5.2×, no spill) |
| turbo3/turbo3 | 16384 | 9.75 tok/s (smaller KV, fits) |
| turbo3/turbo3 | **2048** | **9.38 ± 0.93 tok/s** (recommended: safe at all ctx sizes) |

Full benchmarks + needle tests at:
https://github.com/KaiFelixBennett/gemma4-turboquant-rdna4

### Tested on
- gfx1201 (RDNA4), HIP SDK 7.1, Clang 21, Windows 11
- Gemma-4-31B-it Q4_K_M

### Patch
See `patches/0001-turbo4-hip-graph-safe-fattn.patch` in the benchmarks repo above.

---

## Companion — patch 0002 (same fork)

**Title:** `HIP/Windows: drop unavailable peer-to-peer memcpy path; add cudaEventCreate mapping`

These are Windows + HIP SDK 7.1 build fixes for the fork — not an upstream concern (upstream
doesn't use `cudaEventCreate`, and its peer-memcpy path is guarded differently).

### Changes

**`ggml/src/ggml-cuda/ggml-cuda.cu`** — Remove `cudaMemcpy3DPeerAsync` path

`hipMemcpy3DPeerParms` / `hipMemcpy3DPeerAsync` are not available on Windows HIP SDK 7.1.
The staging-buffer fallback is always safe and already present. Remove the peer path to
fix a linker error on Windows.

**`ggml/src/ggml-cuda/vendors/hip.h`** — Add missing `cudaEventCreate` mapping

```cpp
#define cudaEventCreate hipEventCreate
```

Without this, `llama-server` fails to compile on Windows HIP SDK 7.1 with
`error: use of undeclared identifier 'cudaEventCreate'`.

### Tested on
- gfx1201 (RDNA4), HIP SDK 7.1, Windows 11

### Note
These are build-time fixes for Windows + HIP SDK 7.1. If the peer-to-peer path was
intentional for other platforms, the `#ifdef` guard can be made platform-conditional:
```cpp
#if defined(__HIP_PLATFORM_AMD__) && defined(_WIN32)
    // staging fallback only — hipMemcpy3DPeerAsync unavailable on Windows HIP
#else
    // peer path ...
#endif
```
