# Launch Gemma-4-31B-it with the recommended TurboQuant config.
# Self-contained: calls llama-server.exe directly, no external project or YAML needed.
#
# Usage:
#   .\configs\run_gemma4.ps1 -ModelPath "E:\...\gemma-4-31B-it-Q4_K_M.gguf"
#   .\configs\run_gemma4.ps1 -ModelPath ... -CtxSize 262144      # full 256K

param(
    [Parameter(Mandatory = $true)][string]$ModelPath,
    [string]$BinaryDir = "C:\models\llama-cpp-tq\build\bin",   # default: scripts\setup.ps1 output
    [int]$CtxSize = 131072,                                    # 262144 for full 256K
    [int]$Port = 8080,
    [string]$HIP_PATH = "C:\Program Files\AMD\ROCm\7.1"
)

$ErrorActionPreference = "Stop"

$server = Join-Path $BinaryDir "llama-server.exe"
if (-not (Test-Path $server)) {
    Write-Host "llama-server.exe not found at $server" -ForegroundColor Red
    Write-Host "  Build first (scripts\setup.ps1) or pass -BinaryDir. See docs/BUILD-WINDOWS-HIP.md" -ForegroundColor Yellow
    exit 1
}
if (-not (Test-Path $ModelPath)) {
    Write-Host "Model not found: $ModelPath" -ForegroundColor Red
    exit 1
}

$env:Path = (Join-Path $HIP_PATH "bin") + ";" + $env:Path

# Recommended config: q8_0 keys + turbo4 values, small batch (NOT 16384), single slot.
& $server `
    -m $ModelPath `
    --alias "Gemma-4-31B-it" `
    --host 127.0.0.1 --port $Port `
    --ctx-size $CtxSize `
    --batch-size 2048 --ubatch-size 512 `
    --flash-attn on `
    --cache-type-k q8_0 --cache-type-v turbo4 `
    --parallel 1 `
    --jinja --reasoning-format auto
