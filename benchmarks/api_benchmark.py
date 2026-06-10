"""API-based benchmark — hits the running llama-server at 127.0.0.1:8080.
Does NOT start its own llama.cpp instance. Measures real prefill & decode speed.
Includes PID-lock to prevent concurrent benchmark runs."""

import urllib.request
import urllib.error
import time
import json
import os
import sys
import atexit
import ctypes
import ctypes.wintypes

LOCK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".api_benchmark.lock")
SCRIPT_NAME = os.path.basename(__file__)  # "api_benchmark.py"

def pid_is_alive(pid: int) -> bool:
    """Check if a Windows process with the given PID is still running."""
    kernel32 = ctypes.windll.kernel32
    PROCESS_QUERY_INFORMATION = 0x0400
    h = kernel32.OpenProcess(PROCESS_QUERY_INFORMATION, False, pid)
    if h == 0:
        return False
    code = ctypes.wintypes.DWORD()
    kernel32.GetExitCodeProcess(h, ctypes.byref(code))
    kernel32.CloseHandle(h)
    return code.value == 259  # STILL_ACTIVE

def find_other_benchmark_instances():
    """Scan ALL running processes for other api_benchmark.py instances.
    Returns list of PIDs that are NOT this process."""
    others = []
    my_pid = os.getpid()
    try:
        import subprocess
        out = subprocess.check_output(
            ['wmic', 'process', 'where', 'name="python.exe"', 'get', 'ProcessId,CommandLine'],
            timeout=10, text=True, stderr=subprocess.DEVNULL
        )
        for line in out.splitlines():
            if SCRIPT_NAME in line:
                parts = line.strip().split()
                if parts:
                    try:
                        pid = int(parts[-1])
                        if pid != my_pid and pid_is_alive(pid):
                            others.append(pid)
                    except (ValueError, IndexError):
                        pass
    except Exception:
        pass
    return others

def acquire_lock() -> bool:
    """Try to acquire the PID lock. Returns True if lock acquired, False if another instance is running.
    Uses BOTH lock file AND process scan for defense in depth."""

    # 1. Check lock file first
    if os.path.exists(LOCK_FILE):
        try:
            with open(LOCK_FILE) as f:
                old_pid = int(f.read().strip())
            if pid_is_alive(old_pid):
                print(f"❌ Benchmark läuft bereits (Lock-PID {old_pid}).")
                print(f"   Lock-Datei: {LOCK_FILE}")
                return False
            else:
                print(f"⚠️  Veraltete Lock-Datei (PID {old_pid} ist tot) — wird überschrieben.")
        except (ValueError, FileNotFoundError):
            print("⚠️  Beschädigte Lock-Datei — wird überschrieben.")

    # 2. DEFENSE IN DEPTH: scan for orphan processes without lock file
    orphans = find_other_benchmark_instances()
    if orphans:
        print(f"❌ Benchmark läuft bereits (Orphan-PIDs: {orphans})!")
        print(f"   Diese Prozesse haben KEINE Lock-Datei — kill_terminal hat nur das Terminal gekillt.")
        print(f"   Bitte beende sie manuell: Stop-Process -Id {orphans[0]} -Force")
        return False

    # 3. All clear — write lock
    with open(LOCK_FILE, "w") as f:
        f.write(str(os.getpid()))
    atexit.register(lambda: os.remove(LOCK_FILE) if os.path.exists(LOCK_FILE) else None)
    print(f"🔒 Lock acquired (PID {os.getpid()})")
    return True

if not acquire_lock():
    sys.exit(1)

API = "http://127.0.0.1:8080/v1/chat/completions"
MODEL = "Gemma-4-31B-it GGUF"
CONTEXTS = [2048, 4096, 8192, 16384, 32768, 65536, 131072]
GEN_TOKENS = 256

def fill_prompt(target_tokens):
    """Generate ~target_tokens of filler text."""
    base = "The quick brown fox jumps over the lazy dog. "
    repeated = base * (target_tokens // 6 + 1)
    return repeated[: target_tokens * 4]

results = []

# Load existing results if resuming
OUT_PATH = "results/api_bench_b8192.json"
if os.path.exists(OUT_PATH):
    try:
        with open(OUT_PATH) as f:
            existing = json.load(f)
        if isinstance(existing, list) and existing:
            completed_ctxs = {r["ctx"] for r in existing if "error" not in r}
            results = existing
            print(f"📂 Resuming from {len(existing)} existing results (contexts: {sorted(completed_ctxs)})")
            CONTEXTS = [c for c in CONTEXTS if c not in completed_ctxs]
            print(f"   Remaining: {[f'{c//1024}K' if c >= 1024 else str(c) for c in CONTEXTS]}")
    except Exception:
        pass

for ctx in CONTEXTS:
    label = f"{ctx // 1024}K" if ctx >= 1024 else str(ctx)
    print(f"\n{'='*60}")
    print(f"  Benchmark: {label} context ({ctx} tokens)")
    print(f"{'='*60}")

    prompt = fill_prompt(ctx - GEN_TOKENS - 50)
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": f"{prompt}\n\nWhat is the meaning of 'benchmark'? Answer concisely."}
        ],
        "max_tokens": GEN_TOKENS,
        "temperature": 0.0,
        "stream": False
    }

    try:
        t0 = time.perf_counter()
        req = urllib.request.Request(
            API,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=1800) as resp:
            elapsed = time.perf_counter() - t0
            body = resp.read().decode("utf-8")

        data = json.loads(body)
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        timing = data.get("timings", {})
        pp_ms = timing.get("prompt_ms", 0)
        tg_ms = timing.get("predicted_ms", 0)

        pp_tps = (prompt_tokens / pp_ms * 1000) if pp_ms > 0 else 0
        tg_tps = (completion_tokens / tg_ms * 1000) if tg_ms > 0 else 0

        if pp_tps == 0 and tg_tps == 0:
            pp_tps = prompt_tokens / (elapsed * 0.3) if elapsed > 0 else 0
            tg_tps = completion_tokens / (elapsed * 0.7) if elapsed > 0 else 0

        print(f"  Prompt:   {prompt_tokens:>6} tokens | {pp_tps:>8.1f} t/s")
        print(f"  Generate: {completion_tokens:>6} tokens | {tg_tps:>8.1f} t/s")
        print(f"  Total:    {elapsed:>7.1f}s")

        results.append({
            "ctx": ctx,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "pp_tps": round(pp_tps, 1),
            "tg_tps": round(tg_tps, 1),
            "elapsed_s": round(elapsed, 1)
        })

    except urllib.error.HTTPError as e:
        print(f"  ERROR: HTTP {e.code}: {e.read().decode()[:200]}")
        results.append({"ctx": ctx, "error": f"HTTP {e.code}"})
    except urllib.error.URLError as e:
        print(f"  ERROR: Cannot connect: {e.reason}")
        results.append({"ctx": ctx, "error": "connection refused"})
        break
    except TimeoutError:
        print("  ERROR: Timeout (>30 min)")
        results.append({"ctx": ctx, "error": "timeout"})
    except Exception as e:
        print(f"  ERROR: {e}")
        results.append({"ctx": ctx, "error": str(e)})

    # INCREMENTAL SAVE — write after every level so no data is lost on crash
    out_path = "results/api_bench_b8192.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

# Summary
print(f"\n{'='*60}")
print("  SUMMARY (API-based, batch_size=8192)")
print(f"{'='*60}")
print(f"{'Context':>8} {'PP t/s':>10} {'TG t/s':>10} {'Elapsed':>10}")
print(f"{'─'*8} {'─'*10} {'─'*10} {'─'*10}")
for r in results:
    if "error" in r:
        print(f"{r['ctx']:>8} {'ERROR':>10}: {r['error'][:40]}")
    else:
        print(f"{r['ctx']:>8} {r['pp_tps']:>10.1f} {r['tg_tps']:>10.1f} {r['elapsed_s']:>9.1f}s")

print(f"\nResults saved to: {out_path}")
