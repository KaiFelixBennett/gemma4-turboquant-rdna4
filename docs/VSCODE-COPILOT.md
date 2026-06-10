# Local Gemma-4-31B in VS Code GitHub Copilot — a real 176K-token session

This guide wires the TurboQuant llama.cpp build into **GitHub Copilot Chat as a custom
model**, so VS Code talks to your own GPU instead of a cloud API. Everything below was run
on the AMD Radeon AI PRO R9700 (gfx1201, 32 GB) — including a real agent-mode session that
grew past **176K tokens of context**. Numbers from that live session are labeled as such;
the controlled reference numbers come from llama-bench (see [BENCHMARKS.md](BENCHMARKS.md)).

<p align="center">
  <img src="../assets/vscode-copilot-176k.png" alt="VS Code Copilot Chat using local Gemma-4-31B at ~176K context, with llama-server logs and Task Manager GPU memory" width="100%">
  <br>
  <em>A real Copilot agent session at ~176K context against the local server — and the moment
  we caught the session-state trap: 13.8 GB silently swapped into shared GPU memory (see below).</em>
</p>

---

## 1. Server

Start `llama-server` with the recommended config ([`configs/run_gemma4.ps1`](../configs/run_gemma4.ps1))
**plus three flags that cap llama-server's session state** — without them, long Copilot
sessions with an SWA model silently eat ~15 GB extra (the third config trap, measured below):

```powershell
llama-server `
  -m gemma-4-31B-it-Q4_K_M.gguf `
  --alias "Gemma-4-31B-it" `        # MUST exactly match the model "id" in VS Code (step 2)
  --ctx-size 131072 `               # sweet spot; 262144 loads but decode gets slow when filled
  --batch-size 2048 --ubatch-size 512 `
  --flash-attn on `
  --cache-type-k turbo3 --cache-type-v turbo3 `
  --parallel 1 `
  --ctx-checkpoints 4 `             # default 32 x 234 MiB = ~7.3 GB of SWA checkpoints
  --cache-ram 0 `                   # default 8192 MiB prompt cache
  --kv-unified `                    # A/B tested: removing it HALVED decode (keep it on)
  --jinja --reasoning-format auto
```

## 2. VS Code

Copilot Chat → model picker → **Manage Models… → Custom endpoint** (stored in your profile's
`chatLanguageModels.json`):

```jsonc
{
  "name": "llama.cpp",
  "vendor": "customendpoint",
  "apiKey": "llama.cpp",
  "apiType": "chat-completions",
  "models": [
    {
      "id": "Gemma-4-31B-it",        // MUST equal the server --alias, character for character
      "name": "Gemma-4-31B-it (Local)",
      "url": "http://127.0.0.1:8080/v1",
      "toolCalling": true,
      "vision": false,               // text-only GGUF: no vision projector
      "maxInputTokens": 122880,      // input + output must fit inside --ctx-size
      "maxOutputTokens": 8192
    }
  ]
}
```

Reload the VS Code window after editing. The model appears in the Copilot model picker;
agent mode (tools) works — Gemma-4's tool-calling through llama.cpp is functional but not
as robust as the big cloud models, so expect an occasional retried tool call.

## 3. What to expect (measured)

| Phase | Measured | Notes |
|-------|----------|-------|
| Prefill, fresh context | **735 t/s** | llama-bench pp2048, turbo KV + HIP graphs |
| Per-turn re-prefill at 100–190K depth | 70–124 t/s | live session; Copilot appends ~10K tokens/turn (tool defs + results) → **1.5–3 min per turn** at this depth |
| Decode @ 128K | **9.38 ± 0.93 t/s** | llama-bench, turbo3, `-b 2048` — the controlled reference |
| Decode @ ~176K (live, session state capped) | ~2.3 t/s | live Copilot session — the 32 GB card is at its edge here |
| Decode @ ~187K (live, default session state) | 0.85 t/s | **the trap below in action** |

**The honest ceiling:** ≤128K is the comfortable working zone on a 32 GB card. 176K+ works
and answers correctly, but decode is slow. 262144 (full 256K) *loads* — treat it as a
capability demo, not a daily driver.

Append-style conversations reuse the cached prefix automatically (slot LCP matching — the
logs show `sim_best = 0.9+`), so only each turn's new suffix is prefilled. Gemma's SWA means
`--cache-reuse` itself is unsupported (`cache_reuse is not supported by this context`) — that
line in the log is expected, not an error.

## 4. The third config trap: llama-server session state

During the 176K session, decode collapsed progressively (2.11 t/s at 107K → **0.85 t/s** at
187K) and Task Manager showed **13.8 GB in shared GPU memory** (= system RAM over PCIe) on
top of 29/32 GB dedicated. The cause was not the model or the KV cache — it was
**llama-server's session-state defaults**, which for a 256K SWA model add up fast:

| Server feature | Default | Cost @ 256K (Gemma-4-31B) |
|----------------|---------|---------------------------|
| SWA context checkpoints | 32 per slot | 32 × 234 MiB ≈ **7.3 GB** |
| Prompt cache | 8192 MiB | up to **8 GB** |
| OS / other apps (dwm, VS Code, browser) | — | ~3–4 GB GPU memory |

With `--ctx-checkpoints 4 --cache-ram 0`, shared-memory spill dropped from **13.8 GB to
1.35 GB** and live decode at ~176K recovered from 0.85 to ~2.3 t/s. The checkpoint cap
trade-off: edits deep in the context roll back further (more re-prefill in that case) —
irrelevant for append-style agent sessions.

We also A/B-tested `--no-kv-unified` at ~176K fill: it **halved** decode (2.26 → 1.24 t/s).
Keep `--kv-unified` on.

## 5. Troubleshooting

- **Model missing in picker** → `id` ≠ `--alias`. They must match exactly.
- **Empty responses** → Gemma-4 is a thinking model; with `--reasoning-format auto` content
  arrives in `reasoning_content` + `content`. Copilot handles this; raw API clients must read both.
- **Errors when attaching images** → set `"vision": false`; the Q4_K_M text GGUF has no projector.
- **Decode suddenly ~1–2 t/s at long context** → check Task Manager *shared* GPU memory.
  If it's > ~1 GB, you're paging: cap session state (above), close other GPU apps
  (a browser + VS Code + dwm hold 3–4 GB), or drop `--ctx-size` to 131072.
