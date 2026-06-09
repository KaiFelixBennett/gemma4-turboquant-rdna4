# Verify TurboQuant is Active in llama-server
# Checks binary, config, and runs a quick decode speed test
#
# Usage: .\scripts\verify_turboquant.ps1 [-Port 8080]

param(
    [int]$Port = 8080,
    [switch]$SkipTest
)

$ErrorActionPreference = "Stop"

$projectRoot = "C:\Users\KaiFe\Desktop\gemma4-turboquant-rdna4"
$hermesRoot = "C:\Users\KaiFe\Desktop\hermes-claude-code-local"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  TurboQuant Verification for Gemma-4 on AMD Radeon AI PRO" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check which binary directory is configured
Write-Host "[1/5] Checking hermes_config.gemma.yaml..." -ForegroundColor Yellow

$configPath = Join-Path $hermesRoot "hermes_config.gemma.yaml"
if (Test-Path $configPath) {
    $binaryDir = (Select-String "binary_dir" $configPath).Line.Trim() -replace '^binary_dir:\s*"?([^"]*)"?.*$', '$1'
    $cacheTypeK = (Select-String "cache_type_k" $configPath).Line.Trim() -replace '^cache_type_k:\s*"?([^"]*)"?.*$', '$1'
    $cacheTypeV = (Select-String "cache_type_v" $configPath).Line.Trim() -replace '^cache_type_v:\s*"?([^"]*)"?.*$', '$1'
    
    Write-Host "  binary_dir:    $binaryDir" -ForegroundColor White
    Write-Host "  cache_type_k:  $cacheTypeK" -ForegroundColor White
    Write-Host "  cache_type_v:  $cacheTypeV" -ForegroundColor White
    
    if ($binaryDir -eq "bTurboQuant-gfx1201-turbo4") {
        Write-Host "  ✅ Binary directory is TheTom fork (turbo4)" -ForegroundColor Green
    } elseif ($binaryDir -eq "bTurboQuant-gfx1201") {
        Write-Host "  ⚠️  Binary directory is jagsan-cyber (NO turbo4, broken SWA)" -ForegroundColor Red
        Write-Host "     Update hermes_config.gemma.yaml:" -ForegroundColor Yellow
        Write-Host "       binary_dir: `"bTurboQuant-gfx1201-turbo4`"" -ForegroundColor Yellow
        Write-Host "       cache_type_k: `"q8_0`"" -ForegroundColor Yellow
        Write-Host "       cache_type_v: `"turbo4`"" -ForegroundColor Yellow
    } else {
        Write-Host "  ❓ Unknown binary directory: $binaryDir" -ForegroundColor Yellow
    }
    
    if ($cacheTypeV -eq "turbo4") {
        Write-Host "  ✅ cache_type_v is turbo4" -ForegroundColor Green
    } elseif ($cacheTypeV -eq "q4_0") {
        Write-Host "  ⚠️  cache_type_v is q4_0 (NOT TurboQuant)" -ForegroundColor Red
    } else {
        Write-Host "  ❓ Unknown cache_type_v: $cacheTypeV" -ForegroundColor Yellow
    }
} else {
    Write-Host "  ⚠️  Config not found: $configPath" -ForegroundColor Red
}

# Step 2: Check binary exists and has turbo4 support
Write-Host ""
Write-Host "[2/5] Checking llama-server binary..." -ForegroundColor Yellow

$serverPath = Join-Path $hermesRoot "tools\llama.cpp\bTurboQuant-gfx1201-turbo4\llama-server.exe"
if (Test-Path $serverPath) {
    Write-Host "  ✅ Found: $serverPath" -ForegroundColor Green
    
    # Check DLL size (TheTom fork has ~100MB ggml-hip.dll)
    $dllPath = Join-Path $hermesRoot "tools\llama.cpp\bTurboQuant-gfx1201-turbo4\ggml-hip.dll"
    if (Test-Path $dllPath) {
        $dllSize = [math]::Round((Get-Item $dllPath).Length / 1MB, 1)
        Write-Host "  ggml-hip.dll: $dllSize MB" -ForegroundColor White
        if ($dllSize -gt 80) {
            Write-Host "  ✅ DLL size indicates TurboQuant kernels included" -ForegroundColor Green
        } else {
            Write-Host "  ⚠️  DLL size suggests TurboQuant may NOT be included" -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "  ❌ NOT found: $serverPath" -ForegroundColor Red
    Write-Host "     Build and deploy TheTom fork first. See docs/BUILD-WINDOWS-HIP.md" -ForegroundColor Yellow
}

# Step 3: Check turbo4 in --help
Write-Host ""
Write-Host "[3/5] Checking llama-server --help for turbo4..." -ForegroundColor Yellow

$env:Path = "C:\Program Files\AMD\ROCm\7.1\bin;" + $env:Path

if (Test-Path $serverPath) {
    $helpOutput = & $serverPath --help 2>&1 | Out-String
    if ($helpOutput -match "turbo4") {
        Write-Host "  ✅ turbo4 found in --help output" -ForegroundColor Green
    } else {
        Write-Host "  ❌ turbo4 NOT found in --help output" -ForegroundColor Red
        Write-Host "     This binary does NOT support TurboQuant KV cache" -ForegroundColor Yellow
    }
    if ($helpOutput -match "turbo3") {
        Write-Host "  ✅ turbo3 found in --help output" -ForegroundColor Green
    }
}

# Step 4: Check if server is running
Write-Host ""
Write-Host "[4/5] Checking if llama-server is running..." -ForegroundColor Yellow

$apiUrl = "http://127.0.0.1:$Port/v1/models"
try {
    $response = Invoke-RestMethod -Uri $apiUrl -Method Get -TimeoutSec 3
    Write-Host "  ✅ Server is running on port $Port" -ForegroundColor Green
    Write-Host "  Model: $($response.data[0].id)" -ForegroundColor White
    $serverRunning = $true
} catch {
    Write-Host "  ⚠️  Server is NOT running on port $Port" -ForegroundColor Yellow
    Write-Host "     Start it with: start_gemma.bat or start_gemma_turbo4.ps1" -ForegroundColor Yellow
    $serverRunning = $false
}

# Step 5: Quick decode speed test
if ($serverRunning -and -not $SkipTest) {
    Write-Host ""
    Write-Host "[5/5] Running quick decode speed test (4K context)..." -ForegroundColor Yellow
    
    $testPrompt = ("The quick brown fox jumps over the lazy dog. " * 600)
    $payload = @{
        model = "Gemma-4-31B-it GGUF"
        messages = @(
            @{
                role = "user"
                content = "$testPrompt`n`nWhat is 2+2? Answer with just the number."
            }
        )
        max_tokens = 50
        temperature = 0.0
        stream = $false
    } | ConvertTo-Json -Depth 5
    
    $apiUrl = "http://127.0.0.1:$Port/v1/chat/completions"
    
    try {
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $response = Invoke-RestMethod -Uri $apiUrl -Method Post -Body $payload -ContentType "application/json" -TimeoutSec 120
        $sw.Stop()
        
        $usage = $response.usage
        $promptTokens = $usage.prompt_tokens
        $completionTokens = $usage.completion_tokens
        $totalTime = $sw.ElapsedMilliseconds / 1000
        
        if ($completionTokens -gt 0 -and $totalTime -gt 0) {
            $tps = [math]::Round($completionTokens / $totalTime, 1)
            Write-Host "  Prompt tokens: $promptTokens" -ForegroundColor White
            Write-Host "  Completion tokens: $completionTokens" -ForegroundColor White
            Write-Host "  Total time: $([math]::Round($totalTime, 1))s" -ForegroundColor White
            Write-Host "  Decode speed: ~$tps t/s" -ForegroundColor White
            
            if ($tps -ge 15) {
                Write-Host "  ✅ Decode speed looks healthy ($tps t/s at ~4K context)" -ForegroundColor Green
                Write-Host "     TurboQuant + SWA fix appear to be working!" -ForegroundColor Green
            } elseif ($tps -ge 5) {
                Write-Host "  ⚠️  Decode speed is lower than expected ($tps t/s)" -ForegroundColor Yellow
                Write-Host "     Could be normal for this hardware, or a mild SWA issue." -ForegroundColor Yellow
            } else {
                Write-Host "  ❌ Decode speed is very low ($tps t/s)" -ForegroundColor Red
                Write-Host "     The SWA bug may be present!" -ForegroundColor Red
            }
        }
    } catch {
        Write-Host "  ❌ Test request failed: $_" -ForegroundColor Red
    }
} elseif (-not $serverRunning) {
    Write-Host ""
    Write-Host "[5/5] Skipping decode test (server not running)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Summary" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  To use TurboQuant with Gemma-4, ensure:" -ForegroundColor White
Write-Host "  1. binary_dir: `"bTurboQuant-gfx1201-turbo4`" in hermes_config.gemma.yaml" -ForegroundColor White
Write-Host "  2. cache_type_k: `"q8_0`" (NOT q4_0, NOT turbo4)" -ForegroundColor White
Write-Host "  3. cache_type_v: `"turbo4`" (NOT q4_0)" -ForegroundColor White
Write-Host "  4. Start with: start_gemma.bat (uses hermes_config.gemma.yaml)" -ForegroundColor White
Write-Host ""
Write-Host "  For detailed verification, see docs/VERIFY-TURBOQUANT.md" -ForegroundColor Cyan
Write-Host ""