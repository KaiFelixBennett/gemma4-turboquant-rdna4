# SWA Pattern Verification Script
# Checks if the running llama-server correctly parses Gemma-4's SWA pattern
# Usage: .\scripts\verify_swa.ps1 [-Port 8080]

param(
    [int]$Port = 8080
)

$ErrorActionPreference = "Stop"

$apiUrl = "http://127.0.0.1:$Port/v1/models"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  SWA Pattern Verification for Gemma-4" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Check if server is running
try {
    $response = Invoke-RestMethod -Uri $apiUrl -Method Get -TimeoutSec 5
    Write-Host "Server is running on port $Port" -ForegroundColor Green
    Write-Host "Model: $($response.data[0].id)" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Cannot connect to llama-server on port $Port" -ForegroundColor Red
    Write-Host "  Start the server first, e.g.: .\configs\run_gemma4.ps1 -ModelPath <path-to-gguf>" -ForegroundColor Yellow
    exit 1
}

# Send a test prompt and check timing
Write-Host ""
Write-Host "Running SWA verification test..." -ForegroundColor Yellow
Write-Host "  Sending 4K context prompt and measuring decode speed..." -ForegroundColor Cyan

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
    
    $timing = $response.timings
    $usage = $response.usage
    
    if ($timing) {
        $ppTps = [math]::Round($usage.prompt_tokens / $timing.prompt_ms * 1000, 1)
        $tgTps = [math]::Round($usage.completion_tokens / $timing.predicted_ms * 1000, 1)
        
        Write-Host ""
        Write-Host "  Prompt tokens: $($usage.prompt_tokens)" -ForegroundColor White
        Write-Host "  Completion tokens: $($usage.completion_tokens)" -ForegroundColor White
        Write-Host "  Prefill: $ppTps t/s" -ForegroundColor White
        Write-Host "  Decode:  $tgTps t/s" -ForegroundColor White
        Write-Host ""
        
        # SWA bug detection heuristic
        if ($tgTps -lt 5) {
            Write-Host "  *** WARNING: Decode speed is very low ($tgTps t/s at ~4K context) ***" -ForegroundColor Red
            Write-Host "  This may indicate the SWA bug is present." -ForegroundColor Red
            Write-Host "  Expected: 15-25 t/s at 4K context with correct SWA." -ForegroundColor Yellow
            Write-Host "  If SWA is broken, ALL layers do global attention, causing" -ForegroundColor Yellow
            Write-Host "  severe decode slowdown even at short contexts." -ForegroundColor Yellow
        } elseif ($tgTps -lt 10) {
            Write-Host "  *** CAUTION: Decode speed is lower than expected ***" -ForegroundColor Yellow
            Write-Host "  Expected: 15-25 t/s at 4K context." -ForegroundColor Yellow
            Write-Host "  This could be normal for this hardware, or a mild SWA issue." -ForegroundColor Yellow
        } else {
            Write-Host "  *** GOOD: Decode speed looks healthy ($tgTps t/s) ***" -ForegroundColor Green
            Write-Host "  SWA pattern appears to be working correctly." -ForegroundColor Green
        }
    } else {
        Write-Host "  No timing data in response (server may not support /timings)" -ForegroundColor Yellow
        Write-Host "  Total time: $($sw.ElapsedMilliseconds)ms" -ForegroundColor White
    }
    
} catch {
    Write-Host "ERROR: Request failed: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  For definitive SWA verification, check server startup logs:" -ForegroundColor Cyan
Write-Host "  Look for 'sliding_window_pattern' in the output." -ForegroundColor Cyan
Write-Host "  Correct Gemma-4 pattern: 5 SWA + 1 Global per 6 layers" -ForegroundColor Cyan
Write-Host "  If ALL layers show 'global', the SWA bug is present." -ForegroundColor Red
Write-Host "============================================================" -ForegroundColor Cyan