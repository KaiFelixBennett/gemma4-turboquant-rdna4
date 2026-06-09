# How to Verify TurboQuant is Active

When you start `start_gemma.bat` (or `start_gemma_turbo4.ps1`), you need to confirm that TurboQuant KV cache is actually being used. Here are the methods:

## Method 1: Check Server Startup Logs (Most Reliable)

When llama-server starts, it prints the KV cache configuration. Look for these lines:

```
llama_model_load: cache_type_k = q8_0
llama_model_load: cache_type_v = turbo4
```

**If you see `q4_0` for either K or V, TurboQuant is NOT active.** You're running the old jagsan-cyber build.

**If you see `turbo4` for V, TurboQuant IS active.** ✅

### What the startup output should look like:

```
llama_model_load: model =       17872.02 MiB / 32624.00 MiB
llama_model_load: cache_type_k = q8_0
llama_model_load: cache_type_v = turbo4
llama_model_load: n_ctx = 262144 (262144)
llama_model_load: n_batch = 8192
llama_model_load: n_ubatch = 2048
llama_model_load: flash_attn = 1
llama_model_load: n_gpu_layers = 99
ggml_cuda_init: found 1 CUDA devices:
  Device 0: AMD Radeon AI PRO R9700, compute capability 10.0.1, VMM: 0
```

## Method 2: Check the Binary Directory

The `hermes_config.gemma.yaml` has a `binary_dir` setting:

```yaml
# OLD (broken SWA, no turbo4):
binary_dir: "bTurboQuant-gfx1201"

# NEW (SWA fix + turbo4):
binary_dir: "bTurboQuant-gfx1201-turbo4"
```

Check which directory is being used:

```powershell
# Check the config
Select-String "binary_dir" hermes_config.gemma.yaml

# Should show: binary_dir: "bTurboQuant-gfx1201-turbo4"
```

## Method 3: Check the DLL Size

The TheTom fork's `ggml-hip.dll` is ~100 MB (includes TurboQuant kernels). The jagsan-cyber build's DLL is smaller.

```powershell
# TheTom fork (with TurboQuant)
(Get-Item "tools\llama.cpp\bTurboQuant-gfx1201-turbo4\ggml-hip.dll").Length / 1MB
# ~100 MB

# jagsan-cyber build (without TurboQuant)
(Get-Item "tools\llama.cpp\bTurboQuant-gfx1201\ggml-hip.dll").Length / 1MB
# ~50-60 MB
```

## Method 4: Run the Verification Script

```powershell
cd C:\Users\KaiFe\Desktop\gemma4-turboquant-rdna4
.\scripts\verify_turboquant.ps1
```

This script:
1. Checks which binary directory is configured
2. Checks if `turbo4` appears in `llama-server.exe --help`
3. Runs a quick 4K context test and measures decode speed
4. Reports whether TurboQuant appears to be working

## Method 5: Decode Speed Heuristic

At 4K context with Gemma-4-31B-it Q4_K_M:

| Configuration | Expected Decode Speed |
|---------------|----------------------|
| Broken SWA + q4_0/q4_0 | ~15-20 t/s |
| Fixed SWA + q8_0/turbo4 | ~18-25 t/s |
| Fixed SWA + q8_0/turbo4 at 128K | ~8-15 t/s |
| Broken SWA + q4_0/q4_0 at 128K | **~1.4 t/s** (catastrophic) |

**If decode speed at 128K context is below 5 t/s, the SWA bug is likely present.**

## Method 6: Check SWA Pattern in Server Logs

When llama-server loads Gemma-4, it should print the sliding window attention pattern:

```
llama_model_load: n_layer = 60, n_head = 16, n_head_kv = 8
llama_model_load: sliding_window_pattern = [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, ...]
```

- `0` = global attention (every 6th layer)
- `1` = sliding window attention (5 of every 6 layers)

**If ALL values are `0`, the SWA bug is present.** Every layer would be doing global attention, causing catastrophic decode slowdown.

**Correct pattern for Gemma-4**: `[0, 1, 1, 1, 1, 1]` repeating (1 global + 5 SWA per 6 layers).

## Quick Checklist

| Check | Expected | Broken |
|-------|----------|--------|
| `binary_dir` in config | `bTurboQuant-gfx1201-turbo4` | `bTurboQuant-gfx1201` |
| `cache_type_k` in logs | `q8_0` | `q4_0` |
| `cache_type_v` in logs | `turbo4` | `q4_0` |
| `ggml-hip.dll` size | ~100 MB | ~50-60 MB |
| Decode @ 4K | 18-25 t/s | 15-20 t/s |
| Decode @ 128K | 8-15 t/s | **1.4 t/s** |
| SWA pattern | `[0,1,1,1,1,1]` repeating | All `0`s |