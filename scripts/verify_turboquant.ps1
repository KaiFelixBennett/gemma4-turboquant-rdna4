# Verify a TurboQuant build is active.
# Checks the built binary for turbo support and (optionally) runs a quick decode test
# against a running llama-server. No dependency on any external/private project.
#
# Usage:
#   .\scripts\verify_turboquant.ps1 [-BinaryDir <path>] [-Port 8080] [-SkipTest]

param(
    [string]$BinaryDir = "C:\models\llama-cpp-tq\build\bin",   # default: scripts\setup.ps1 output
    [int]$Port = 8080,
    [string]$HIP_PATH = "C:\Program Files\AMD\ROCm\7.1",
    [switch]$SkipTest
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  TurboQuant build verification" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

$serverPath = Join-Path $BinaryDir "llama-server.exe"

# Step 1: binary exists
Write-Host "[1/4] Checking llama-server binary..." -ForegroundColor Yellow
if (Test-Path $serverPath) {
    Write-Host "  OK  Found: $serverPath" -ForegroundColor Green
    $dllPath = Join-Path $BinaryDir "ggml-hip.dll"
    if (Test-Path $dllPath) {
        $dllSize = [math]::Round((Get-Item $dllPath).Length / 1MB, 1)
        Write-Host "  ggml-hip.dll: $dllSize MB (TheTom fork's TurboQuant build is ~100 MB)" -ForegroundColor White
    }
} else {
    Write-Host "  !!  NOT found: $serverPath" -ForegroundColor Red
    Write-Host "      Build first (scripts\setup.ps1) or pass -BinaryDir. See docs/BUILD-WINDOWS-HIP.md" -ForegroundColor Yellow
    exit 1
}

# Step 2: turbo types in --help
Write-Host ""
Write-Host "[2/4] Checking --help for turbo cache types..." -ForegroundColor Yellow
$env:Path = (Join-Path $HIP_PATH "bin") + ";" + $env:Path
$helpOutput = & $serverPath --help 2>&1 | Out-String
if ($helpOutput -match "turbo4") {
    Write-Host "  OK  turbo4 available" -ForegroundColor Green
} else {
    Write-Host "  !!  turbo4 NOT found - this binary lacks TurboQuant KV support" -ForegroundColor Red
}
if ($helpOutput -match "turbo3") { Write-Host "  OK  turbo3 available" -ForegroundColor Green }

# Step 3: server running?
Write-Host ""
Write-Host "[3/4] Checking for a running llama-server on port $Port..." -ForegroundColor Yellow
$serverRunning = $false
try {
    $response = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/models" -Method Get -TimeoutSec 3
    Write-Host "  OK  Server running. Model: $($response.data[0].id)" -ForegroundColor Green
    $serverRunning = $true
} catch {
    Write-Host "  --  No server on port $Port (start one to run the decode test)" -ForegroundColor Yellow
}

# Step 4: quick decode test
if ($serverRunning -and -not $SkipTest) {
    Write-Host ""
    Write-Host "[4/4] Quick decode test (~4K context)..." -ForegroundColor Yellow
    $testPrompt = ("The quick brown fox jumps over the lazy dog. " * 600)
    $payload = @{
        messages = @(@{ role = "user"; content = "$testPrompt`n`nWhat is 2+2? Answer with just the number." })
        max_tokens = 50
        temperature = 0.0
        stream = $false
    } | ConvertTo-Json -Depth 5
    try {
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $response = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/chat/completions" -Method Post -Body $payload -ContentType "application/json" -TimeoutSec 120
        $sw.Stop()
        $ct = $response.usage.completion_tokens
        $secs = $sw.ElapsedMilliseconds / 1000
        if ($ct -gt 0 -and $secs -gt 0) {
            $tps = [math]::Round($ct / $secs, 1)
            Write-Host "  Decode speed: ~$tps t/s at ~4K context" -ForegroundColor White
            if ($tps -ge 15) {
                Write-Host "  OK  Healthy (expected ~18-25 t/s at 4K)" -ForegroundColor Green
            } elseif ($tps -ge 5) {
                Write-Host "  ?   Lower than expected - could be normal, or a mild SWA issue" -ForegroundColor Yellow
            } else {
                Write-Host "  !!  Very low - the SWA bug may be present" -ForegroundColor Red
            }
        }
    } catch {
        Write-Host "  !!  Test request failed: $_" -ForegroundColor Red
    }
} elseif (-not $serverRunning) {
    Write-Host ""
    Write-Host "[4/4] Skipping decode test (no server running)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Done. For the full verification guide see docs/VERIFY-TURBOQUANT.md" -ForegroundColor Cyan
Write-Host ""
