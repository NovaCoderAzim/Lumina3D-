<#
Runs the Lumina3D backend in FULL REAL mode using the Python 3.11 venv
(.venv311) that has pycolmap + open3d + torch(CUDA) + ultralytics.

Usage:  powershell -ExecutionPolicy Bypass -File scripts\run_backend_real.ps1
#>
$env:USE_MOCK_VISION = "false"
$env:USE_MOCK_RECONSTRUCTION = "false"
$env:USE_MOCK_AI = "false"
$env:USE_MOCK_ANALYTICS = "false"
$env:USE_GPU = "false"                 # pycolmap pip wheel: CPU SIFT only
$env:MATCHER_TYPE = "sequential"       # video frames are ordered -> faster
$env:DENSE_RECONSTRUCTION_ENABLED = "true"   # real dense MVS via CUDA COLMAP CLI
$env:COLMAP_PATH = (Join-Path (Join-Path $PSScriptRoot "..") "tools\colmap-cuda\bin\colmap.exe")
$env:MASK_DYNAMIC_OBJECTS = "true"           # mask moving objects (vehicles/people)
$env:VISION_TARGET_FRAMES = "80"
$env:VISION_CANDIDATE_STRIDE = "2"
$env:VISION_MIN_SHARPNESS = "15.0"
$env:PYTHONUTF8 = "1"

Push-Location (Join-Path $PSScriptRoot "..")
try {
    & ".venv311\Scripts\python.exe" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
}
finally {
    Pop-Location
}
