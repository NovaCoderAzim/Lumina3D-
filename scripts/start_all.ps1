<#
Lumina3D — start EVERYTHING with one command.

Launches the two servers you need and nothing else:
  1. Backend  : FastAPI (real mode, .venv311) on http://127.0.0.1:8000
  2. Frontend : Vite dev server (real backend) on http://localhost:5173

Each runs in its own window. The browser opens to the app once the
frontend is up.

Usage:  powershell -ExecutionPolicy Bypass -File scripts\start_all.ps1
#>

$root = Join-Path $PSScriptRoot ".."
$root = (Resolve-Path $root).Path

Write-Host "Starting Lumina3D..." -ForegroundColor Green
Write-Host "  backend  -> http://127.0.0.1:8000  (docs at /docs)" -ForegroundColor Gray
Write-Host "  frontend -> http://localhost:5173" -ForegroundColor Gray

# 1. Backend (real mode) in its own window.
Start-Process powershell -ArgumentList @(
    "-NoExit", "-ExecutionPolicy", "Bypass",
    "-File", (Join-Path $root "scripts\run_backend_real.ps1")
) -WorkingDirectory $root

# 2. Frontend in its own window.
Start-Process powershell -ArgumentList @(
    "-NoExit", "-ExecutionPolicy", "Bypass",
    "-File", (Join-Path $root "scripts\run_frontend.ps1")
) -WorkingDirectory $root

# Give the frontend a moment, then open the browser.
Start-Sleep -Seconds 8
Start-Process "http://localhost:5173"

Write-Host "Both servers launching in separate windows. App: http://localhost:5173" -ForegroundColor Green
