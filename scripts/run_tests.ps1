<#
Runs the Lumina3D integration + unit tests in full mock mode.
Usage:  powershell -ExecutionPolicy Bypass -File scripts\run_tests.ps1
#>
$env:USE_MOCK_VISION = "true"
$env:USE_MOCK_RECONSTRUCTION = "true"
$env:USE_MOCK_AI = "true"
$env:USE_MOCK_ANALYTICS = "true"
$env:PYTHONUTF8 = "1"

Push-Location (Join-Path $PSScriptRoot "..")
try {
    python -m pytest tests/ -v
}
finally {
    Pop-Location
}
