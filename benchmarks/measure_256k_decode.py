#!/usr/bin/env python3
"""
measure_256k_decode.py — measure real-world decode speed at very long context.

Fills the KV cache with a realistic-length filler document, then streams a
generation and reports tokens/s at that context depth.

Usage:
    # Server must be running at -c 262144 first:
    python benchmarks/measure_256k_decode.py --base-url http://127.0.0.1:8080/v1 --ctx 200000
"""
import argparse
import json
import sys
import time
import urllib.request

FILLER_TEMPLATE = (
    "The following is a large technical document about software engineering, "
    "distributed systems, and machine learning. "
)

FILLER_SENTENCES = [
    "Software engineers must balance performance, reliability, and maintainability in large-scale systems.",
    "Distributed consensus protocols such as Raft and Paxos underpin modern cloud infrastructure.",
    "Gradient descent optimization forms the backbone of neural network training algorithms.",
    "Containerization with Docker and Kubernetes has transformed deployment pipelines.",
    "Retrieval-augmented generation combines neural language models with external knowledge bases.",
    "The attention mechanism enables transformers to model long-range token dependencies efficiently.",
    "Asynchronous programming patterns improve throughput in I/O-bound server applications.",
    "Static analysis tools catch bugs early and enforce coding conventions across large codebases.",
    "Microservices architectures decompose monolithic applications into independently deployable units.",
    "Memory-bandwidth is the dominant bottleneck for large language model inference at long context.",
    "Quantization reduces model precision while preserving most of the accuracy for downstream tasks.",
    "Flash Attention computes attention in tiles to avoid materializing the full N×N attention matrix.",
    "AMD's RDNA4 architecture delivers high compute density at competitive power efficiency.",
    "KV cache compression allows longer context lengths without proportional VRAM growth.",
]


def count_tokens(base_url: str, text: str) -> int:
    url = f"{base_url.rstrip('/')}/tokenize"
    payload = json.dumps({"content": text}).encode()
    req = urllib.request.Request(url, data=payload, method="POST",
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return len(json.load(resp)["tokens"])


def build_filler(target_tokens: int, base_url: str) -> str:
    """Build a filler document targeting roughly target_tokens tokens."""
    # calibrate: ~17 tokens per sentence (from previous needle test)
    n_sentences = target_tokens // 17
    sentences = [FILLER_SENTENCES[i % len(FILLER_SENTENCES)] for i in range(n_sentences)]
    return FILLER_TEMPLATE + " ".join(sentences)


def stream_at_depth(base_url: str, filler: str, question: str, max_tokens: int) -> tuple[int, float]:
    """Send filler + question, stream response, return (n_tokens, avg_tps)."""
    prompt = filler + f"\n\nQuestion: {question}\nAnswer:"
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
    tokens = 0
    first_t = None
    with urllib.request.urlopen(req, timeout=600) as resp:
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
            content = delta.get("content") or delta.get("reasoning_content") or ""
            if content:
                if first_t is None:
                    first_t = time.perf_counter()
                tokens += 1
                elapsed = time.perf_counter() - first_t
                tps = tokens / elapsed if elapsed > 0 else 0
                print(f"\r  [{tokens:4d} tokens | {tps:5.1f} tok/s | {elapsed:5.1f}s]",
                      end="", flush=True)
    print()
    elapsed = time.perf_counter() - (first_t or time.perf_counter())
    avg_tps = tokens / elapsed if elapsed > 0 else 0
    return tokens, avg_tps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--ctx", type=int, default=100000, help="Target filler context tokens")
    ap.add_argument("--max-tokens", type=int, default=200)
    args = ap.parse_args()

    print(f"Building filler document (~{args.ctx} tokens)...", flush=True)
    filler = build_filler(args.ctx, args.base_url)

    # Count actual tokens
    try:
        actual = count_tokens(args.base_url, filler)
        print(f"Actual filler tokens: {actual:,}")
    except Exception as e:
        actual = args.ctx
        print(f"Tokenizer check failed ({e}), using estimate: {actual:,}")

    question = "What is the main bottleneck for large language model inference at long context, according to this document?"

    print(f"\nStreaming generation at ~{actual:,} token context...\n")
    n_tok, avg_tps = stream_at_depth(args.base_url, filler, question, args.max_tokens)
    print(f"\nContext depth: ~{actual:,} tokens")
    print(f"Generated:     {n_tok} tokens")
    print(f"Average speed: {avg_tps:.2f} tok/s")


if __name__ == "__main__":
    main()
