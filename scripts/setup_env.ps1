param(
    [switch]$SkipInstall = $false,
    [switch]$SkipGdal = $false
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

function Invoke-Step {
    param([string]$Message)
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Install-Packages {
    param(
        [string[]]$Packages
    )
    & python -m pip install @Packages
}

Invoke-Step "Updating pip tooling"
Install-Packages @("--upgrade", "pip", "setuptools", "wheel")

if (-not $SkipInstall) {
    Invoke-Step "Installing CUDA-enabled PyTorch and torchvision"
    Install-Packages @("torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu118")

    Invoke-Step "Installing core geospatial and experiment dependencies"
    Install-Packages @(
        "ultralytics",
        "rasterio",
        "geopandas",
        "pycocotools",
        "requests",
        "scipy",
        "scikit-image"
    )

    if (-not $SkipGdal) {
        try {
            Invoke-Step "Attempting to install GDAL (optional on Windows if rasterio already works)"
            Install-Packages @("gdal")
        }
        catch {
            Write-Warning "GDAL installation failed. Rasterio-based Stage 0/1 scripts still work; install GDAL manually later if needed."
        }
    }
}

Invoke-Step "Verifying critical imports and CUDA visibility"
@'
import importlib
import json
import sys

report = {}
optional = ["gdal"]
required = ["torch", "torchvision", "ultralytics", "rasterio", "geopandas", "pycocotools", "scipy", "skimage"]

for name in required + optional:
    try:
        module = importlib.import_module(name)
        report[name] = {"ok": True, "version": getattr(module, "__version__", "n/a")}
    except Exception as exc:
        report[name] = {"ok": False, "error": str(exc)}

try:
    import torch
    report["torch_cuda"] = {
        "available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "device_count": int(torch.cuda.device_count()),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
except Exception as exc:
    report["torch_cuda"] = {"available": False, "error": str(exc)}

print(json.dumps(report, indent=2))
failed_required = [name for name in required if not report[name]["ok"]]
if failed_required:
    print(f"Missing required imports: {failed_required}", file=sys.stderr)
    sys.exit(1)
'@ | python -

Invoke-Step "Exporting requirements.txt"
& python -m pip freeze | Set-Content -Encoding utf8 requirements.txt

Write-Host "Environment setup finished. requirements.txt has been refreshed." -ForegroundColor Green
