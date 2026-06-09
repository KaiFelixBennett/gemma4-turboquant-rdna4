<#
.SYNOPSIS
    Single-command setup for Gemma-4-31B TurboQuant on AMD RDNA4 (gfx1201).

.DESCRIPTION
    Clones TheTom/llama-cpp-turboquant at the tested commit, applies the two patches
    in patches/ (HIP-graph-safe Flash-Attention + Windows HIP build fixes), and builds
    for gfx1201 with HIP graphs enabled. Then prints the exact server command to run.

    This is a thin wrapper around scripts/build_turboquant.ps1. Run it from anywhere:
        .\scripts\setup.ps1 -ModelPath "E:\path\to\gemma-4-31B-it-Q4_K_M.gguf"

.PARAMETER RepoDir
    Where to clone/build the llama.cpp fork. Default: C:\models\llama-cpp-tq

.PARAMETER ModelPath
    Path to the Gemma-4-31B-it Q4_K_M GGUF. Used only to print the run command.

.PARAMETER HIP_PATH
    HIP SDK install path. Default: C:\Program Files\AMD\ROCm\7.1
#>
param(
    [string]$RepoDir   = "C:\models\llama-cpp-tq",
    [string]$ModelPath = "<path-to>\gemma-4-31B-it-Q4_K_M.gguf",
    [string]$GPU_TARGET = "gfx1201",
    [string]$HIP_PATH  = "C:\Program Files\AMD\ROCm\7.1",
    [string]$Commit    = "7d9715f"
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Gemma-4 TurboQuant RDNA4 - single-command setup" -ForegroundColor Cyan
Write-Host "  Target: $GPU_TARGET | Commit: $Commit" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# Clone + patch + build (HIP graphs ON) via the build script.
& (Join-Path $here "build_turboquant.ps1") `
    -RepoDir $RepoDir `
    -Commit  $Commit `
    -GPU_TARGET $GPU_TARGET `
    -HIP_PATH $HIP_PATH

if ($LASTEXITCODE -ne 0) {
    Write-Host "Setup failed during build. See output above." -ForegroundColor Red
    exit 1
}

$serverExe = Join-Path $RepoDir "build\bin\llama-server.exe"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  READY. Run Gemma-4-31B with the recommended config:" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  & `"$serverExe`" ``"                                  -ForegroundColor White
Write-Host "      -m `"$ModelPath`" ``"                              -ForegroundColor White
Write-Host "      --alias `"Gemma-4-31B-it`" ``"                     -ForegroundColor White
Write-Host "      --host 127.0.0.1 --port 8080 ``"                   -ForegroundColor White
Write-Host "      --ctx-size 131072 ``      # 262144 for full 256K"  -ForegroundColor White
Write-Host "      --batch-size 2048 --ubatch-size 512 ``  # CRITICAL: not 16384" -ForegroundColor Yellow
Write-Host "      --flash-attn on ``"                                -ForegroundColor White
Write-Host "      --cache-type-k q8_0 --cache-type-v turbo4 ``"      -ForegroundColor White
Write-Host "      --parallel 1 --jinja"                              -ForegroundColor White
Write-Host ""
Write-Host "  Verify:  .\scripts\verify_turboquant.ps1"             -ForegroundColor Cyan
Write-Host "  Needle:  python benchmarks\needle_test.py --base-url http://127.0.0.1:8080/v1 --label q8_0-turbo4" -ForegroundColor Cyan
Write-Host ""
