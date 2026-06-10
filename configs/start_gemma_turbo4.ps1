# Gemma-4-31B TurboQuant Launch Script
# Starts llama-server with TheTom fork + q8_0-K + turbo4-V configuration
#
# Prerequisites:
#   1. Build TheTom/llama-cpp-turboquant for gfx1201 (see docs/BUILD-WINDOWS-HIP.md)
#   2. Place the built binaries in tools/llama.cpp/bTurboQuant-gfx1201-turbo4/
#   3. Update $ConfigPath below if needed

param(
    [string]$ConfigPath = "$PSScriptRoot\..\configs\hermes_config.gemma.turbo4.yaml"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

# Resolve config path
if (-not [System.IO.Path]::IsPathRooted($ConfigPath)) {
    $ConfigPath = Join-Path $repoRoot $ConfigPath
}

if (-not (Test-Path $ConfigPath)) {
    Write-Host "ERROR: Config not found: $ConfigPath" -ForegroundColor Red
    Write-Host "  Copy configs/hermes_config.gemma.turbo4.yaml and adjust paths." -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Gemma-4-31B with TurboQuant (q8_0-K + turbo4-V)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Config: $ConfigPath" -ForegroundColor White
Write-Host "  KV Cache: q8_0-K + turbo4-V (asymmetric)" -ForegroundColor Green
Write-Host ""
Write-Host "  IMPORTANT: This requires TheTom/llama-cpp-turboquant fork!" -ForegroundColor Yellow
Write-Host "  See docs/BUILD-WINDOWS-HIP.md for build instructions." -ForegroundColor Yellow
Write-Host ""

# Use the shared launcher
$launcher = Join-Path $repoRoot "start_llamacpp.ps1"
if (Test-Path $launcher) {
    & $launcher -ConfigPath $ConfigPath
} else {
    Write-Host "ERROR: start_llamacpp.ps1 not found at $launcher" -ForegroundColor Red
    Write-Host "  This script expects to be inside the hermes-claude-code-local repo." -ForegroundColor Yellow
    exit 1
}