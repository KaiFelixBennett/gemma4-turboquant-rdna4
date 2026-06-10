# Automated Build Script for TheTom llama-cpp-turboquant on Windows + HIP/ROCm
# Targets: AMD Radeon AI PRO R9700 / RX 9070 XT (gfx1201, RDNA4)
#
# Usage: .\scripts\build_turboquant.ps1
# Prerequisites: HIP SDK 7.1, VS 2022 Build Tools, CMake, Ninja, Git

param(
    [string]$RepoDir = "C:\models\llama-cpp-tq",
    [string]$Branch = "feature/turboquant-kv-cache",
    [string]$Commit = "7d9715f",
    [string]$GPU_TARGET = "gfx1201",
    [string]$HIP_PATH = "C:\Program Files\AMD\ROCm\7.1",
    [switch]$SkipClone,
    [switch]$SkipPatches,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  TurboQuant llama.cpp Build Script for Windows + HIP/ROCm" -ForegroundColor Cyan
Write-Host "  Target: $GPU_TARGET (RDNA4)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Verify prerequisites
Write-Host "[1/6] Checking prerequisites..." -ForegroundColor Yellow

$hipBin = Join-Path $HIP_PATH "bin"
# Merge the machine + user PATH so the VS toolchain / RC compiler are found even when this
# script is launched from a thin shell (avoids "No CMAKE_RC_COMPILER could be found").
$env:PATH = "$hipBin;C:\Program Files\CMake\bin;" + `
    [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + `
    [System.Environment]::GetEnvironmentVariable("Path","User")
$env:HIP_PATH = $HIP_PATH

# Check HIP
$hipcc = Get-Command "hipcc.exe" -ErrorAction SilentlyContinue
if (-not $hipcc) {
    Write-Host "ERROR: hipcc not found. Install HIP SDK 7.1 and set HIP_PATH." -ForegroundColor Red
    Write-Host "  Download from: https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html" -ForegroundColor Red
    exit 1
}
Write-Host "  hipcc: $($hipcc.Source)" -ForegroundColor Green

# Check CMake
$cmake = Get-Command "cmake.exe" -ErrorAction SilentlyContinue
if (-not $cmake) {
    Write-Host "ERROR: cmake not found. Install CMake 4.3.1+" -ForegroundColor Red
    exit 1
}
Write-Host "  cmake: $($cmake.Source)" -ForegroundColor Green

# Check Ninja
$ninja = Get-Command "ninja.exe" -ErrorAction SilentlyContinue
if (-not $ninja) {
    Write-Host "WARNING: ninja not found. Install with: pip install ninja" -ForegroundColor Yellow
}
Write-Host "  ninja: $($ninja.Source)" -ForegroundColor Green

# Check clang
$clang = Get-Command "clang.exe" -ErrorAction SilentlyContinue
if (-not $clang) {
    Write-Host "ERROR: clang not found. It should come with HIP SDK." -ForegroundColor Red
    exit 1
}
Write-Host "  clang: $($clang.Source)" -ForegroundColor Green

# Step 2: Clone repo
if (-not $SkipClone) {
    Write-Host ""
    Write-Host "[2/6] Cloning TheTom/llama-cpp-turboquant..." -ForegroundColor Yellow
    if (Test-Path $RepoDir) {
        Write-Host "  Repo already exists at $RepoDir" -ForegroundColor Green
        Push-Location $RepoDir
        git fetch origin
        git checkout $Commit
        Pop-Location
    } else {
        # Full clone (not --branch/--depth) so the pinned commit is always reachable.
        git clone https://github.com/TheTom/llama-cpp-turboquant.git $RepoDir
        Push-Location $RepoDir
        git fetch origin $Commit
        git checkout $Commit
        Pop-Location
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: could not check out pinned commit $Commit." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "[2/6] Skipping clone (requested)" -ForegroundColor Yellow
}

# Step 3: Apply Windows HIP patches
if (-not $SkipPatches) {
    Write-Host ""
    Write-Host "[3/6] Applying Windows HIP patches..." -ForegroundColor Yellow
    $patchesDir = Join-Path $PSScriptRoot "..\patches"
    
    if (Test-Path $patchesDir) {
        Get-ChildItem -Path $patchesDir -Filter "*.patch" | Sort-Object Name | ForEach-Object {
            Write-Host "  Applying $($_.Name)..." -ForegroundColor Cyan
            Push-Location $RepoDir
            git apply --check $_.FullName
            if ($LASTEXITCODE -ne 0) {
                Write-Host "  ERROR: patch does not apply cleanly: $($_.Name)" -ForegroundColor Red
                Pop-Location; exit 1
            }
            git apply $_.FullName
            if ($LASTEXITCODE -ne 0) {
                Write-Host "  ERROR: git apply failed: $($_.Name)" -ForegroundColor Red
                Pop-Location; exit 1
            }
            Pop-Location
        }
    } else {
        Write-Host "  No patch files found in $patchesDir" -ForegroundColor Yellow
        Write-Host "  Manual patches may be required. See docs/BUILD-WINDOWS-HIP.md" -ForegroundColor Yellow
    }
} else {
    Write-Host "[3/6] Skipping patches (requested)" -ForegroundColor Yellow
}

# Step 4: Configure CMake
Write-Host ""
Write-Host "[4/6] Configuring CMake..." -ForegroundColor Yellow

$buildDir = Join-Path $RepoDir "build"
if ($Clean -and (Test-Path $buildDir)) {
    Write-Host "  Cleaning build directory..." -ForegroundColor Cyan
    Remove-Item -Recurse -Force $buildDir
}

Push-Location $RepoDir

cmake -S . -B build -G Ninja `
    -DGPU_TARGETS=$GPU_TARGET `
    -DGGML_HIP=ON `
    -DGGML_HIP_GRAPHS=ON `
    -DGGML_CUDA_FA_ALL_QUANTS=ON `
    -DCMAKE_C_COMPILER=clang `
    -DCMAKE_CXX_COMPILER=clang++ `
    -DCMAKE_BUILD_TYPE=Release

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: CMake configuration failed!" -ForegroundColor Red
    Pop-Location
    exit 1
}

# Step 5: Build
Write-Host ""
Write-Host "[5/6] Building..." -ForegroundColor Yellow

cmake --build build --config Release

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Build failed!" -ForegroundColor Red
    Pop-Location
    exit 1
}

Pop-Location

# Step 6: Verify
Write-Host ""
Write-Host "[6/6] Verifying build..." -ForegroundColor Yellow

$serverExe = Join-Path $RepoDir "build\bin\llama-server.exe"
if (Test-Path $serverExe) {
    Write-Host "  llama-server.exe found: $serverExe" -ForegroundColor Green
    
    # Check for turbo types
    $helpOutput = & $serverExe --help 2>&1 | Out-String
    if ($helpOutput -match "turbo3" -and $helpOutput -match "turbo4") {
        Write-Host "  turbo3 and turbo4 cache types: AVAILABLE" -ForegroundColor Green
    } else {
        Write-Host "  WARNING: turbo3/turbo4 cache types NOT found in help output!" -ForegroundColor Yellow
        Write-Host "  The build may not include TurboQuant support." -ForegroundColor Yellow
    }
} else {
    Write-Host "  ERROR: llama-server.exe not found at $serverExe" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  BUILD SUCCESSFUL!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Binary: $serverExe" -ForegroundColor Cyan
Write-Host ""
Write-Host "  To run Gemma-4-31B with TurboQuant:" -ForegroundColor Cyan
Write-Host "  llama-server.exe \" -ForegroundColor White
Write-Host "    -m <path-to-model>\ " -ForegroundColor White
Write-Host "    --cache-type-k q8_0 \" -ForegroundColor White
Write-Host "    --cache-type-v turbo4 \" -ForegroundColor White
Write-Host "    -ngl 99 -c 131072 -b 2048 -ub 512 -fa on --jinja" -ForegroundColor White
Write-Host ""
Write-Host "  Recommended: q8_0-K + turbo4-V for highest fidelity (needle 9/9)." -ForegroundColor Yellow
Write-Host "  turbo3/turbo3 also works on gfx1201 (no NaN) for max context + speed." -ForegroundColor Yellow