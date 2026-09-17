<#
Runs the DRISHTI-3D backend in full mock mode (no GPU/COLMAP/weights).
Usage:  powershell -ExecutionPolicy Bypass -File scripts\run_backend.ps1
#>
$env:USE_MOCK_VISION = "true"
$env:USE_MOCK_RECONSTRUCTION = "true"
$env:USE_MOCK_AI = "true"
$env:USE_MOCK_ANALYTICS = "true"
$env:PYTHONUTF8 = "1"

Push-Location (Join-Path $PSScriptRoot "..")
try {
    python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
}
finally {
    Pop-Location
}
