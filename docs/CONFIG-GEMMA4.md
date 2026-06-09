# Gemma-4-31B Configuration Reference

## Model Specifications

| Parameter | Value |
|-----------|-------|
| Model | Gemma-4-31B-it |
| Architecture | Dense transformer with hybrid SWA |
| Parameters | 31B |
| Layers | 60 |
| Attention Heads | 32 Q / 16 KV (GQA) |
| Head Dim | 256 (global) / 128 (SWA) |
| Context Length | 262,144 (256K) |
| SWA Window | 1,024 |
| SWA Pattern | 5 SWA + 1 global per 6 layers |
| Thinking Mode | Native `<\|channel\|>thought` |
| Quantization | Q4_K_M (17.46 GiB) |

## Recommended Server Configuration

### With TurboQuant (TheTom fork)

```yaml
# hermes_config.gemma.turbo4.yaml
providers:
  local-llama-cpp:
    name: "llama.cpp"
    base_url: "http://127.0.0.1:8080/v1"
    api_key: "llama.cpp"
    api_mode: "chat_completions"
    default_model: "Gemma-4-31B-it GGUF"
    model: "Gemma-4-31B-it GGUF"
    models:
      - "Gemma-4-31B-it GGUF"

model:
  path: 'E:\Coding\custom-rag\data\models\gemma-4-31b-it\gemma-4-31B-it-Q4_K_M.gguf'
  default: "Gemma-4-31B-it GGUF"
  chat_template: ''
  backend: "hip"
  binary_dir: "bTurboQuant-gfx1201"
  provider: "custom:local-llama-cpp"
  base_url: "http://127.0.0.1:8080/v1"
  api_key: "llama.cpp"
  context_length: 131072   # 128K for agentic coding; 262144 for full 256K
  batch_size: 2048         # CRITICAL: large batch spills VRAM at long context
  ubatch_size: 512
  # TURBOQUANT: Asymmetric K/V for Q4_K_M models
  # q8_0-K preserves attention routing quality
  # turbo4-V compresses value cache 3.8x with +0.23% PPL
  cache_type_k: "q8_0"
  cache_type_v: "turbo4"
  parallel_slots: 1
  flash_attn: true
  # Google's official sampling parameters for Gemma-4
  temperature: 1.0
  top_p: 0.95
  top_k: 64
  min_p: 0.0
  presence_penalty: 0.0
  repeat_penalty: 1.0
  reasoning: on
  reasoning_budget: 2048
  speculative_type: "none"
  speculative_draft_tokens: ""
```

### Without TurboQuant (jagsan-cyber fork — baseline)

```yaml
# hermes_config.gemma.yaml (current baseline)
cache_type_k: "q4_0"
cache_type_v: "q4_0"
# All other settings same as above
```

## Sampling Parameters

Google's official recommendations for Gemma-4 instruct mode:

| Parameter | Value | Notes |
|-----------|-------|-------|
| temperature | 1.0 | Gemma-4 is trained for temp=1.0 |
| top_p | 0.95 | Standard nucleus sampling |
| top_k | 64 | Gemma-4 specific |
| min_p | 0.0 | Not needed with top_k=64 |
| presence_penalty | 0.0 | Community consensus for coding |
| repeat_penalty | 1.0 | No repeat penalty needed |

## KV Cache Comparison

Measured on this hardware (gfx1201). "Same-top-p" is the KL-divergence agreement vs the
f16 baseline at `-c 512`; "Needle" is long-context retrieval at 8K–33K (see docs/QUALITY.md).
VRAM is the measured dedicated GPU memory at 128K idle with `-b 2048`.

| Config (K/V) | Same-top-p @512 | Needle 8K–33K | VRAM @128K | Notes |
|--------------|-----------------|---------------|-----------|-------|
| f16 / f16 | 100% | — | doesn't fit | baseline reference |
| q8_0 / q8_0 | 87.2% | — | ~21 GB | excellent, larger KV |
| **q8_0 / turbo4** | **76.5%** | **9/9** | 21.6 GB | **recommended default** |
| turbo4 / turbo4 | 74.3% | — | 21.6 GB | good; symmetric works here |
| q8_0 / turbo3 | 47.5% | — | 20.6 GB | drifts @512, fine at long ctx |
| turbo3 / turbo3 | 41.0% | **9/9** | 20.6 GB | max context/speed; lossless retrieval |

**Key insights:**
- On Q4_K_M models, K precision protects attention routing via softmax; V compression is
  nearly free. `q8_0-K + turbo4-V` is the safe high-fidelity default.
- The low `turbo3` same-top-p @512 is a **regime artifact**: at 512 tokens the context is
  smaller than Gemma's 1024 SWA window, so the long-context KV path is never exercised. The
  needle test (9/9) shows `turbo3/turbo3` is lossless for long-context retrieval.
- `turbo3` and symmetric `turbo4` run fine on gfx1201 — **no NaNs** in any KLD or needle run.
  Earlier "turbo3 NaN on AMD" reports did not reproduce here.

## SWA Pattern for Gemma-4

Gemma-4-31B has 60 layers with this repeating pattern:

```
Layer 0:  SWA (sliding window 1024)
Layer 1:  SWA
Layer 2:  SWA
Layer 3:  SWA
Layer 4:  SWA
Layer 5:  Global (full attention)
Layer 6:  SWA
...
Layer 59: Global
```

Pattern: `[SWA, SWA, SWA, SWA, SWA, Global]` × 10 = 50 SWA + 10 Global

This means only **1/6 of layers** need full O(n²) attention, making long context feasible with correct SWA parsing.