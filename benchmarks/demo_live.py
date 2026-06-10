#!/usr/bin/env python3
"""
demo_live.py — Live terminal demo for the Reddit GIF/video.

Shows:
 1. Server start at 256K context, VRAM stats
 2. A streaming generation session showing tokens/s
 3. The "before" stats for comparison

Usage:
    # Start the server first (see step 0 below), then run:
    python benchmarks/demo_live.py --base-url http://127.0.0.1:8080/v1

Record with ScreenToGif or Xbox Game Bar (Win+G).
"""
import argparse
import json
import sys
import time
import urllib.request
from datetime import datetime

BANNER = r"""
╔══════════════════════════════════════════════════════════════════╗
║  Gemma-4-31B at 256K context · AMD Radeon AI PRO R9700 (RDNA4)  ║
║  TurboQuant KV cache · Two HIP-graph patches · $1,400 GPU       ║
╚══════════════════════════════════════════════════════════════════╝
"""

BEFORE = """
┌─ BEFORE (broken config: -b 16384, VRAM spill at 128K) ──────────┐
│  Decode @ 128K:  1.28 tok/s  (unusable for agentic coding)       │
│  VRAM @ 128K:   23.40 GB dedicated + 1.15 GB spill  (OOM soon)  │
│  256K context:  OOM — does not load                              │
└──────────────────────────────────────────────────────────────────┘
"""

AFTER_HDR = """
┌─ AFTER (patched: -b 2048, turbo3 KV, HIP-graph-safe FA) ────────┐
│  Decode @ 128K:  ~10 tok/s   (✓ usable for agentic coding)       │
│  VRAM @ 256K:   22.88 GB dedicated  (~9 GB free)                 │
│  256K context:  loads cleanly ✓                                  │
└──────────────────────────────────────────────────────────────────┘
"""

DEMO_PROMPT = (
    "You are running on a $1,400 AMD Radeon AI PRO R9700 with full 256K native context "
    "enabled via TurboQuant KV cache compression. "
    "In one paragraph, explain what TurboQuant is and why 256K context matters for "
    "agentic coding assistants."
)


def check_server(base_url: str, timeout: float = 3.0) -> bool:
    try:
        r = urllib.request.urlopen(f"{base_url.rstrip('/')}/health".replace("/v1", ""), timeout=timeout)
        return r.status == 200
    except Exception:
        return False


def stream_completion(base_url: str, prompt: str, max_tokens: int = 300) -> tuple[str, float]:
    """Stream a chat completion, return (full_text, avg_tps)."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = json.dumps({
        "model": "Gemma-4-31B-it",
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "top_p": 0.95,
    }).encode()
    req = urllib.request.Request(url, data=payload, method="POST",
                                  headers={"Content-Type": "application/json"})
    text = ""
    tokens = 0
    t0 = time.perf_counter()
    first_token_t = None
    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            # Handle thinking model: check both content and reasoning_content
            content = delta.get("content") or delta.get("reasoning_content") or ""
            if content:
                if first_token_t is None:
                    first_token_t = time.perf_counter()
                tokens += 1
                text += content
                elapsed = time.perf_counter() - (first_token_t or t0)
                tps = tokens / elapsed if elapsed > 0 else 0
                print(f"\r  [{tokens:4d} tokens | {tps:5.1f} tok/s | {elapsed:5.1f}s elapsed]",
                      end="", flush=True)
    print()  # newline after progress
    elapsed = time.perf_counter() - (first_token_t or t0)
    avg_tps = tokens / elapsed if elapsed > 0 else 0
    return text, avg_tps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--max-tokens", type=int, default=300)
    args = ap.parse_args()

    print(BANNER)
    print(f"  Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Server: {args.base_url}")
    print()

    # Check server
    print("  Checking server... ", end="", flush=True)
    if not check_server(args.base_url):
        print("NOT READY")
        print()
        print("  Start the server first:")
        print(r'  llama-server.exe -m gemma-4-31B-it-Q4_K_M.gguf -ngl 99 \')
        print(r'    --ctx-size 262144 -b 2048 -ub 512 -fa on \')
        print(r'    --cache-type-k turbo3 --cache-type-v turbo3 --port 8080')
        sys.exit(1)
    print("OK")
    print()

    print(BEFORE)
    time.sleep(1.5)
    print(AFTER_HDR)
    time.sleep(1.5)

    print("=" * 68)
    print("  LIVE DEMO — streaming generation at 256K context")
    print("=" * 68)
    print(f"\n  Prompt: {DEMO_PROMPT[:80]}...")
    print()
    print("  Generating", end="", flush=True)

    full_text, avg_tps = stream_completion(args.base_url, DEMO_PROMPT, args.max_tokens)

    print()
    print("─" * 68)
    print(f"  Average: {avg_tps:.1f} tok/s")
    print()
    print("  Model output:")
    # Word-wrap at 66 chars
    import textwrap
    for line in textwrap.wrap(full_text.strip(), 66):
        print(f"    {line}")
    print()
    print("─" * 68)
    print("  https://github.com/KaiFelixBennett/gemma4-turboquant-rdna4")
    print("─" * 68)


if __name__ == "__main__":
    main()
