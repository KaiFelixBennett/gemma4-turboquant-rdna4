#!/usr/bin/env python3
"""Needle-in-a-haystack long-context retrieval test for a running llama-server.

Measures whether a model can retrieve a small fact ("needle") embedded at a given
depth inside a large filler context ("haystack"). This mirrors the downstream-task
methodology used by Google Research to evaluate TurboQuant KV-cache compression,
and is the metric that actually matters for agentic coding with large context.

The script assumes a llama-server (OpenAI-compatible) is already listening.
It uses only the Python standard library (urllib) so it needs no dependencies.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

# A distinctive needle that cannot be guessed from the filler text.
NEEDLE_TEMPLATE = (
    "Important confidential note: the access key for vault number 7 "
    "is QUASAR-{code}. Remember this exact key."
)
QUESTION = (
    "What is the access key for vault number 7? "
    "Answer with only the key, nothing else."
)

# Bland, varied filler so the model cannot shortcut via repetition.
FILLER_SENTENCES = [
    "The logistics report for the regional warehouse was filed on schedule.",
    "Quarterly maintenance of the cooling units proceeded without incident.",
    "The committee reviewed the budget projections for the coming fiscal year.",
    "Several shipping containers were rerouted due to weather conditions.",
    "The training manual was updated to reflect the new safety procedures.",
    "Inventory counts were reconciled against the central database records.",
    "The night shift completed the audit of the eastern storage facility.",
    "A new vendor agreement was signed for office supply procurement.",
    "The analytics dashboard displayed steady traffic throughout the week.",
    "Routine calibration of the sensors was performed by the technical team.",
]


def post_json(url: str, payload: dict, timeout: int = 600) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def count_tokens(base_url: str, text: str) -> int:
    try:
        res = post_json(f"{base_url}/tokenize", {"content": text}, timeout=120)
        return len(res.get("tokens", []))
    except Exception:
        # Fallback: rough estimate if /tokenize is unavailable.
        return max(1, len(text) // 4)


def build_haystack(target_tokens: int, base_url: str) -> str:
    """Build filler text of approximately target_tokens tokens."""
    lines = []
    i = 0
    # Approximate first, then trust the server to use whatever length results.
    # Each filler line "(N) sentence." measures ~17 tokens on the Gemma-4 tokenizer,
    # so target_tokens // 17 yields roughly target_tokens of actual prompt length.
    approx_lines = max(1, target_tokens // 17)
    while i < approx_lines:
        lines.append(f"({i + 1}) {FILLER_SENTENCES[i % len(FILLER_SENTENCES)]}")
        i += 1
    return "\n".join(lines)


def run_one(base_url: str, target_tokens: int, depth: float, code: str,
            max_tokens: int) -> dict:
    needle = NEEDLE_TEMPLATE.format(code=code)
    haystack = build_haystack(target_tokens, base_url)
    lines = haystack.split("\n")
    insert_at = int(len(lines) * depth)
    lines.insert(insert_at, needle)
    context = "\n".join(lines)

    prompt = (
        "You are given a long document. Read it carefully and then answer the "
        "question based only on the document.\n\n"
        f"<document>\n{context}\n</document>\n\n{QUESTION}"
    )

    n_ctx_tokens = count_tokens(base_url, prompt)

    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    t0 = time.time()
    resp = post_json(f"{base_url}/v1/chat/completions", payload, timeout=900)
    dt = time.time() - t0
    choice = resp["choices"][0]
    msg = choice.get("message", {})
    answer = (msg.get("content") or "").strip()
    reasoning = (msg.get("reasoning_content") or "").strip()
    finish = choice.get("finish_reason", "")
    # A thinking model may place the final key in content, or only reach it inside
    # reasoning if truncated. Search both so we score genuine retrieval ability.
    haystack_for_match = f"{answer}\n{reasoning}"
    passed = code in haystack_for_match
    expected = f"QUASAR-{code}"
    shown = (answer or reasoning).replace("\n", " ").strip()
    return {
        "target_tokens": target_tokens,
        "prompt_tokens": n_ctx_tokens,
        "depth": depth,
        "expected": expected,
        "answer": shown[:120],
        "finish": finish,
        "passed": passed,
        "seconds": round(dt, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--label", default="config", help="label for this run (e.g. q8_0/turbo4)")
    ap.add_argument("--sizes", default="8000,16000,32000",
                    help="comma-separated target context token counts")
    ap.add_argument("--depths", default="0.1,0.5,0.9",
                    help="comma-separated needle depths (0..1)")
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    depths = [float(d) for d in args.depths.split(",") if d.strip()]

    # Distinct 4-digit code per (size,depth) so caching cannot leak answers.
    base_code = 4700
    results = []
    print(f"=== Needle test: {args.label} ===")
    print(f"{'size':>7} {'tokens':>7} {'depth':>6} {'pass':>5} {'sec':>6}  answer")
    idx = 0
    for size in sizes:
        for depth in depths:
            code = str(base_code + idx)
            idx += 1
            try:
                r = run_one(args.base_url, size, depth, code, args.max_tokens)
            except urllib.error.URLError as e:
                print(f"{size:>7}  ERROR  {depth:>6}  request failed: {e}")
                continue
            r["label"] = args.label
            results.append(r)
            mark = "PASS" if r["passed"] else "FAIL"
            print(f"{size:>7} {r['prompt_tokens']:>7} {depth:>6.2f} {mark:>5} "
                  f"{r['seconds']:>6.1f}  {r['answer']}")

    n = len(results)
    passed = sum(1 for r in results if r["passed"])
    rate = (passed / n * 100) if n else 0.0
    print(f"--- {args.label}: {passed}/{n} passed ({rate:.0f}%) ---")

    # Append JSONL for later aggregation.
    with open("needle_results.jsonl", "a", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
