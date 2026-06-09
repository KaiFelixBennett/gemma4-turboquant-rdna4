# Build Guide: TheTom llama-cpp-turboquant on Windows 11 + HIP SDK 7.1 (gfx1201/RDNA4)

> **Complete, tested, reproducible** build instructions for AMD Radeon AI PRO R9700 / RX 9070 XT (gfx1201, RDNA4, 32GB VRAM) on Windows 11.

## What This Builds

The [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) fork (`feature/turboquant-kv-cache` branch) with:

- ✅ **TurboQuant KV cache** — turbo2, turbo3, turbo4 cache types (3.8-5.1x compression)
- ✅ **SWA bug fix** — Correct boolean array parsing for Gemma-4 hybrid attention
- ✅ **Flash Attention** — All quantization types supported (`GGML_CUDA_FA_ALL_QUANTS=ON`)
- ✅ **HIP/ROCm 7.1** — Native gfx1201 (RDNA4) support, no `HSA_OVERRIDE_GFX_VERSION` needed

## Prerequisites

| Tool | Version | Install Command | Notes |
|------|---------|----------------|-------|
| Git | 2.x | Pre-installed on Windows | |
| Python | 3.10+ | Windows Store / python.org | For Ninja via pip |
| VS 2022 Community | v17.14+ | `winget install Microsoft.VisualStudio.2022.Community` | **Must be 2022, NOT 2019** |
| CMake | 4.3.1+ | `winget install Kitware.CMake` | |
| Ninja | 1.13+ | `pip install ninja` | pip version more reliable than winget |
| HIP SDK | 7.1 | Manual from [AMD](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html) | ~1.6GB, installs to `C:\Program Files\AMD\ROCm\7.1` |

### VS 2022 Community Install

Install with the **"Desktop development with C++"** workload:

```powershell
winget install Microsoft.VisualStudio.2022.Community --override `
    "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools `
    --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    --add Microsoft.VisualStudio.Component.Windows11SDK.22621"
```

Or use the Visual Studio Installer GUI and select:
- ✅ Desktop development with C++
- ✅ MSVC v143 - VS 2022 C++ x64/x86 build tools
- ✅ Windows 11 SDK (10.0.22621.0)

### HIP SDK 7.1 Verification

```powershell
$env:Path = "C:\Program Files\AMD\ROCm\7.1\bin;" + $env:Path
hipcc --version
# Should show: HIP version 7.1.51803, Clang 21.0.0

hipinfo
# Should show: gfx1201, Wave Size: 32, NO_VMM
```

## Step-by-Step Build

### 1. Clone TheTom's Fork

```powershell
cd C:\Users\KaiFe\Desktop  # or wherever you want it
git clone --branch feature/turboquant-kv-cache https://github.com/TheTom/llama-cpp-turboquant.git
cd llama-cpp-turboquant
```

### 2. Apply Windows HIP Patches

There are **2 required patches** for Windows + HIP. Both are in the `patches/` directory of this project.

#### Patch 1: Remove Peer-to-Peer Memcpy (ggml-cuda.cu)

HIP on Windows does not support `cudaMemcpy3DPeerParms` / `hipMemcpy3DPeerParms`. The peer-to-peer branch must be removed so the code always uses the staging buffer fallback.

**File**: `ggml/src/ggml-cuda/ggml-cuda.cu`  
**Function**: `ggml_cuda_copy2d_across_devices()` (around line 1930)

**Before** (broken on Windows HIP):
```cpp
    if (info.peer_access[src_device][dst_device]) {
        cudaMemcpy3DPeerParms p = {};
        p.srcPtr = make_cudaPitchedPtr(src_ptr, src_stride, width, height);
        p.dstPtr = make_cudaPitchedPtr(dst_ptr, dst_stride, width, height);
        p.extent = make_cudaExtent(width, height, 1);
        p.srcDevice = src_device;
        p.dstDevice = dst_device;
        CUDA_CHECK(cudaMemcpy3DPeerAsync(&p, stream));
    } else {
        // staging buffer fallback
        ...
    }
```

**After** (always use fallback):
```cpp
    // Peer-to-peer memcpy removed: hipMemcpy3DPeerParms not available on Windows HIP
    // Always use staging buffer fallback
    {
        // staging buffer fallback
        ...
    }
```

#### Patch 2: Add cudaEventCreate Mapping (hip.h)

**File**: `ggml/src/ggml-cuda/vendors/hip.h`  
**Location**: After the existing `#define cudaFuncAttributeMaxDynamicSharedMemorySizeBytes` line

**Add**:
```cpp
#define cudaEventCreate hipEventCreate
```

### 3. Configure CMake

From **PowerShell** (not Git Bash — Ninja PATH issues):

```powershell
$env:HIP_PATH = "C:\Program Files\AMD\ROCm\7.1"
$env:CMAKE_PREFIX_PATH = "C:\Program Files\AMD\ROCm\7.1"
$env:Path = "C:\Program Files\AMD\ROCm\7.1\bin;C:\Program Files\CMake\bin;" + `
    [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + `
    [System.Environment]::GetEnvironmentVariable("Path","User")

# Use the pip-installed ninja (more reliable than winget)
$ninjaPath = (Get-Command ninja.exe -ErrorAction Stop).Source
Write-Host "Using Ninja: $ninjaPath"

cmake -S . -B build -G Ninja `
    -DGPU_TARGETS=gfx1201 `
    -DGGML_HIP=ON `
    -DGGML_CUDA_FA_ALL_QUANTS=ON `
    -DCMAKE_C_COMPILER=clang `
    -DCMAKE_CXX_COMPILER=clang++ `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_MAKE_PROGRAM="$ninjaPath"
```

### 4. Build

```powershell
cmake --build build --config Release
```

Build time: ~10-30 minutes depending on CPU. Expect 643+ compilation units.

### 5. Verify Build

```powershell
# Check that turbo types are available
build\bin\llama-server.exe --help | Select-String "turbo"

# Should show: turbo3, turbo4 in cache-type-k/v options

# Quick test with Gemma-4
$env:Path = "C:\Program Files\AMD\ROCm\7.1\bin;" + $env:Path
build\bin\llama-cli.exe `
    -m "E:\Coding\custom-rag\data\models\gemma-4-31b-it\gemma-4-31B-it-Q4_K_M.gguf" `
    -ngl 99 -c 4096 -fa on `
    --cache-type-k q8_0 --cache-type-v turbo4 `
    -n 50 -p "Hello, I am a language model running on"
```

Expected output should show:
- `AMD Radeon AI PRO R9700` detected
- `gfx1201` architecture
- `32624 MiB` VRAM
- `cache_type_k = q8_0, cache_type_v = turbo4`

### 6. Deploy

Copy the built binaries to your project:

```powershell
$dest = "C:\Users\KaiFe\Desktop\hermes-claude-code-local\tools\llama.cpp\bTurboQuant-gfx1201-turbo4"
New-Item -ItemType Directory -Force -Path $dest

# Copy essential files
Copy-Item "build\bin\llama-server.exe" $dest
Copy-Item "build\bin\llama-cli.exe" $dest
Copy-Item "build\bin\llama-bench.exe" $dest
Copy-Item "build\bin\ggml-hip.dll" $dest
Copy-Item "build\bin\ggml-base.dll" $dest
Copy-Item "build\bin\ggml-cpu.dll" $dest
Copy-Item "build\bin\ggml.dll" $dest
Copy-Item "build\bin\llama.dll" $dest
Copy-Item "build\bin\llama-common.dll" $dest
```

## Known Issues & Gotchas

### 1. VS 2019 Does NOT Work
The `common/` library uses C++17/20 `<functional>` headers that VS 2019's v14.28 doesn't support. **Must use VS 2022.**

### 2. HIP SDK 7.1 Device Math on Windows
MSVC's `<corecrt_math.h>` declares `fabsf`, `fmaxf`, `expf` as `__inline` host-only functions. HIP's clang refuses to call them from `__device__` code.

**Fix**: Add `-xhip -include __clang_hip_runtime_wrapper.h` to compile flags for all `.cu` files in `ggml/src/ggml-hip/CMakeLists.txt`.

### 3. HIP SDK 7.1 Complex Builtins Need min/max
`__clang_cuda_complex_builtins.h` uses bare `min`/`max` which aren't defined in device scope on Windows.

**Fix**: In `ggml/src/ggml-cuda/vendors/hip.h`, add before `<hip/hip_runtime.h>`:
```cpp
#include <algorithm>
using std::min;
using std::max;
```

### 4. M_PI Not Defined with Clang + MSVC Headers
TurboQuant C code uses `M_PI` which MSVC only defines if `_USE_MATH_DEFINES` is set.

**Fix**: In `ggml/src/ggml-turbo-quant.c`, add after includes:
```c
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
```

### 5. Cross-DLL Symbol Visibility
llama.cpp builds as multiple DLLs on Windows. TurboQuant globals shared across DLLs need proper `__declspec` decoration.

**Fix**: Create API functions with `GGML_API` decoration for:
- `turbo3_cpu_wht_group_size` (ggml-base → ggml-cpu)
- `g_innerq_scale_inv_host[]`, `turbo_innerq_needs_tensor_update()`, `turbo_innerq_mark_tensor_updated()` (ggml-hip → llama)

### 6. Ninja PATH Not Available in Git Bash
`winget install Ninja-build.Ninja` adds to Windows PATH but Git Bash doesn't pick it up until new terminal.

**Fix**: `pip install ninja` provides a Python-managed ninja that's always on PATH.

### 7. D>=576 Tile FA Kernels Exceed HIP Local Memory
The HIP CMakeLists excludes `fattn-tile` instances for D=576 and D=640 (exceed 65536 byte local memory limit). But the dispatch code still references them, causing linker errors.

**Fix**: Guard the dispatch cases with `#ifdef GGML_USE_HIP` → `GGML_ABORT(...)`.

### 8. HIP_PATH Trailing Space
When setting `HIP_PATH` in cmd.exe, beware of trailing spaces:
```powershell
# BAD: trailing space after 7.1
set HIP_PATH=C:\Program Files\AMD\ROCm\7.1 
# GOOD: no trailing space
set HIP_PATH=C:\Program Files\AMD\ROCm\7.1
```

### 9. RDNA 4 (gfx1201) Target — It Works!
HIP SDK 7.1 (clang 21) includes gfx1201 support natively. No `HSA_OVERRIDE_GFX_VERSION` needed.

### 10. turbo3 / symmetric turbo4 on gfx1201 — both work here
Earlier community reports claimed `turbo3` produces NaN on AMD HIP and that symmetric
`turbo4/turbo4` is catastrophic on Q4_K_M. **Neither reproduced on this gfx1201 (RDNA4)
card.** In our measurements:
- `turbo3/turbo3` produced no NaNs across all KLD and needle runs and scored **9/9** on the
  long-context needle test (lossless retrieval).
- symmetric `turbo4/turbo4` scored KLD same-top-p 74.3% (vs 76.5% for `q8_0/turbo4`) — close,
  not catastrophic.

The safe high-fidelity default is still **`q8_0-K + turbo4-V`** (it protects attention
routing via 8-bit keys), but `turbo3/turbo3` is a valid choice for maximum context + speed.
See docs/QUALITY.md for the full study.

## Recommended Configuration for Gemma-4-31B-it Q4_K_M

```yaml
cache_type_k: "q8_0"    # 8-bit keys preserve attention routing (softmax is K-sensitive)
cache_type_v: "turbo4"  # highest-fidelity TurboQuant level (needle 9/9)
context_length: 131072  # 128K for agentic coding; 262144 for full 256K
batch_size: 2048        # CRITICAL: large batch spills VRAM at long context
ubatch_size: 512
flash_attn: true
parallel_slots: 1
```

## Build Verification Checklist

- [ ] `hipcc --version` shows HIP 7.1.x and Clang 21.x
- [ ] `hipinfo` shows gfx1201, Wave Size: 32
- [ ] CMake configure succeeds with `GPU_TARGETS=gfx1201`
- [ ] Build completes with 643+ compilation units
- [ ] `llama-server.exe --help | Select-String "turbo"` shows turbo3, turbo4
- [ ] Quick test with Gemma-4 shows AMD Radeon AI PRO R9700 detected
- [ ] Quick test shows `cache_type_k = q8_0, cache_type_v = turbo4`
- [ ] Decode speed at 4K context is >15 t/s (SWA fix working)

## References

- [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) — llama.cpp fork with TurboQuant
- [TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus) — TurboQuant Python prototype
- [TurboQuant Paper (ICLR 2026)](https://arxiv.org/abs/2504.19874) — 4.25-bit KV cache with near-zero quality loss
- [llama.cpp issue #21394](https://github.com/ggml-org/llama.cpp/issues/21394) — SWA bug report