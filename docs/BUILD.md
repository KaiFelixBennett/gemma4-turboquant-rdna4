# Build Guide: TheTom llama-cpp-turboquant on Windows + HIP/ROCm (gfx1201)

Complete step-by-step guide for building llama.cpp with TurboQuant KV cache support on Windows 11 with AMD Radeon AI PRO R9700 / RX 9070 XT (gfx1201, RDNA4).

## Prerequisites

| Tool | Version | Install | Notes |
|------|---------|---------|-------|
| Git | 2.x | Pre-installed | |
| Python | 3.10+ | Windows Store | For turboquant_plus prototype |
| VS 2022 Build Tools | v143 | `winget install Microsoft.VisualStudio.2022.BuildTools` | **MUST be 2022, not 2019** |
| CMake | 4.3.1+ | `winget install Kitware.CMake` | |
| Ninja | latest | `pip install ninja` | pip version more reliable than winget |
| HIP SDK | 7.1 | Manual from AMD | ~1.6GB, installs to `C:\Program Files\AMD\ROCm\7.1` |

### VS 2022 Build Tools Install

```powershell
winget install Microsoft.VisualStudio.2022.BuildTools --override `
    "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools `
    --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    --add Microsoft.VisualStudio.Component.Windows11SDK.22621"
```

### HIP SDK Verification

```powershell
set PATH=C:\Program Files\AMD\ROCm\7.1\bin;%PATH%
hipcc --version
hipinfo
# Should show: gfx1201, Wave Size: 32
```

## Step-by-Step Build

### 1. Clone TheTom's Fork

```powershell
cd C:\models  # or wherever you keep models
git clone --branch feature/turboquant-kv-cache https://github.com/TheTom/llama-cpp-turboquant.git llama-cpp-tq
cd llama-cpp-tq
```

### 2. Apply Windows HIP Patches

There are **9 documented gotchas** for building on Windows + HIP. See `patches/` directory for ready-made patches.

#### Gotcha #1: VS 2019 does NOT work
The `common/` library uses C++17/20 `<functional>` headers that VS 2019's v14.28 doesn't support. **Must use VS 2022.**

#### Gotcha #2: HIP SDK 7.1 device math broken on Windows
MSVC's `<corecrt_math.h>` declares `fabsf`, `fmaxf`, `expf` as `__inline` host-only functions. HIP's clang refuses to call them from `__device__` code.

**Fix**: Add `-xhip -include __clang_hip_runtime_wrapper.h` to compile flags for all `.cu` files:

```cmake
# In ggml/src/ggml-hip/CMakeLists.txt
if (WIN32)
    set(HIP_WIN_FLAGS "-xhip -include __clang_hip_runtime_wrapper.h")
    set_source_files_properties(${GGML_SOURCES_ROCM} PROPERTIES COMPILE_FLAGS "${HIP_WIN_FLAGS}")
endif()
```

#### Gotcha #3: HIP SDK 7.1 complex builtins need min/max
`__clang_cuda_complex_builtins.h` uses bare `min`/`max` which aren't defined in device scope on Windows.

**Fix**: In `ggml/src/ggml-cuda/vendors/hip.h`, add before `<hip/hip_runtime.h>`:
```cpp
#include <algorithm>
using std::min;
using std::max;
```

#### Gotcha #4: M_PI not defined with clang + MSVC headers
The turbo-quant C code uses `M_PI` which MSVC only defines if `_USE_MATH_DEFINES` is set.

**Fix**: In `ggml/src/ggml-turbo-quant.c`, add after includes:
```c
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
```

#### Gotcha #5: Cross-DLL symbol visibility (dllexport/dllimport)
llama.cpp builds as multiple DLLs on Windows. TurboQuant globals shared across DLLs need proper `__declspec` decoration.

**Fix**: Create API functions with `GGML_API` decoration for:
- `turbo3_cpu_wht_group_size` (ggml-base → ggml-cpu)
- `g_innerq_scale_inv_host[]`, `turbo_innerq_needs_tensor_update()`, `turbo_innerq_mark_tensor_updated()` (ggml-hip → llama)

#### Gotcha #6: Ninja PATH not available in Git Bash
`winget install Ninja-build.Ninja` adds to Windows PATH but Git Bash doesn't pick it up until new terminal.

**Fix**: `pip install ninja` provides a Python-managed ninja that's always on PATH.

#### Gotcha #7: D>=576 tile FA kernels exceed HIP local memory
The HIP CMakeLists excludes `fattn-tile` instances for D=576 and D=640 (exceed 65536 byte local memory limit). But the dispatch code still references them, causing linker errors.

**Fix**: Guard the dispatch cases with `#ifdef GGML_USE_HIP` → `GGML_ABORT(...)`.

#### Gotcha #8: HIP_PATH trailing space
When setting `HIP_PATH` in cmd.exe, beware of trailing spaces:
```powershell
# BAD: trailing space after 7.1
set HIP_PATH=C:\Program Files\AMD\ROCm\7.1 
# GOOD: no trailing space
set HIP_PATH=C:\Program Files\AMD\ROCm\7.1
```

#### Gotcha #9: RDNA 4 (gfx1201) target — it works!
HIP SDK 7.1 (clang 21) includes gfx1201 support natively. No `HSA_OVERRIDE_GFX_VERSION` needed.

### 3. Build

From **x64 Native Tools Command Prompt for VS 2022**:

```powershell
cd C:\models\llama-cpp-tq

set PATH=C:\Program Files\AMD\ROCm\7.1\bin;C:\Program Files\CMake\bin;%PATH%
set HIP_PATH=C:\Program Files\AMD\ROCm\7.1

cmake -S . -B build -G Ninja `
    -DGPU_TARGETS=gfx1201 `
    -DGGML_HIP=ON `
    -DGGML_CUDA_FA_ALL_QUANTS=ON `
    -DCMAKE_C_COMPILER=clang `
    -DCMAKE_CXX_COMPILER=clang++ `
    -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release
```

### 4. Verify Build

```powershell
# Check that turbo types are available
build\bin\llama-server.exe --help | Select-String "turbo"

# Should show: turbo3, turbo4 in cache-type-k/v options

# Quick test with Qwen2.5-7B
build\bin\llama-cli.exe `
    -m C:\models\qwen2.5-7b-instruct-q4_k_m.gguf `
    -ngl 99 -c 2048 -fa on `
    --cache-type-k q8_0 --cache-type-v turbo4 `
    -n 100 -p "Hello, I am a language model running on"
```

### 5. Run Gemma-4-31B with TurboQuant

```powershell
build\bin\llama-server.exe `
    -m E:\Coding\custom-rag\data\models\gemma-4-31b-it\gemma-4-31B-it-Q4_K_M.gguf `
    --alias "Gemma-4-31B-it GGUF" `
    --host 127.0.0.1 --port 8080 `
    --ctx-size 131072 `
    --batch-size 2048 --ubatch-size 512 `
    --flash-attn on `
    --cache-type-k q8_0 `
    --cache-type-v turbo4 `
    --parallel 1 `
    --reasoning on --reasoning-budget 2048 `
    --jinja
```

## Troubleshooting

### "turbo3" or "turbo4" not in cache-type options
The build didn't include TurboQuant. Verify `GGML_HIP=ON` and that `ggml-turbo-quant.c` is in the build.

### NaN outputs with turbo3 on Q4_K_M
Earlier community reports claimed this; it **did not reproduce on gfx1201 (RDNA4)** —
`turbo3/turbo3` ran cleanly across all our KLD and needle tests (needle 9/9). If you do see
NaNs on a different AMD architecture, fall back to `q8_0-K + turbo4-V`.

### Build fails with "functional:1259:16: error"
You're using VS 2019. **Must use VS 2022.**

### Build fails with "undefined reference to turbo3_cpu_wht_group_size"
Cross-DLL symbol visibility issue. See Gotcha #5.

### hipinfo shows gfx1201 but build fails
Make sure `HIP_PATH` has no trailing space (Gotcha #8) and `GPU_TARGETS=gfx1201` is set.

## References

- [TheTom/turboquant_plus — Windows RDNA4 Setup Guide](https://github.com/TheTom/turboquant_plus/blob/main/docs/windows-rdna4-setup.md)
- [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) — llama.cpp fork with TurboQuant
- [TurboQuant Paper (ICLR 2026)](https://arxiv.org/abs/2504.19874)