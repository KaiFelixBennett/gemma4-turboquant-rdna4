# Cold-start HIP-graph capture check (turbo KV)

Confirms the warmup-safety assumption behind
[patch 0001](../../patches/0001-turbo4-hip-graph-safe-fattn.patch): with
`GGML_HIP_GRAPHS=ON`, a **genuinely cold** context (no pre-warm) brings up turbo KV
decode without the original capture crash. Redacted excerpt from `llama-cli -v`;
full raw log not committed (2.5 MB of per-token debug spam).

## Run

```
llama-cli -m <gemma-4-31B-it-Q4_K_M.gguf> \
  -ngl 99 -fa on -ctk q8_0 -ctv turbo4 \
  -c 4096 -b 2048 -ub 512 \
  --no-warmup -no-cnv --temp 0 -n 32 -p "..." -v
```

`--no-warmup` is the point: it removes llama's pre-warm pass, so the first real
eval at each size is the user's. Build = TheTom fork `73eb521` + patch 0001,
`GGML_HIP=ON GGML_HIP_GRAPHS=ON`, gfx1201.

## Environment (from the log)

```
print_info: arch              = gemma4
print_info: model type        = 31B
llama_context: n_ctx          = 4096
llama_context: n_batch        = 2048
llama_context: n_ubatch       = 512
llama_context: flash_attn     = enabled
llama_prepare_model_devices: using device ROCm0 (AMD Radeon AI PRO R9700) - 32472 MiB free
```

## The ordering that answers the warmup question

The first eval at a new size runs eager; capture only starts on the next
property-stable eval. So `warmup complete` lands **before** the captured decode
graph is reused:

```
D ggml_backend_cuda_graph_compute: CUDA graph warmup complete
D CUDA Graph id 42 reused
The   D CUDA Graph id 42 reused      <- captured decode + token output
 is   D CUDA Graph id 42 reused
 capital  D CUDA Graph id 42 reused
```

## No capture-time faults

`grep -c` over the full log:

| marker | count |
|---|---|
| `FLASH_ATTN_EXT failed` | 0 |
| `operation not permitted when stream is capturing` | 0 |
| `out of memory` | 0 |
| `failed to allocate` | 0 |

The original bug crashed on the **first** decode step, so a cold-start regression
would surface on token 1. It doesn't.
