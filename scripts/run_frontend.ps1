<#
Starts the Lumina3D frontend dev server (Vite) in REAL-backend mode.
Talks to the backend at http://127.0.0.1:8000 via the /api proxy.

Usage:  powershell -ExecutionPolicy Bypass -File scripts\run_frontend.ps1
Then open http://localhost:5173
#>
Push-Location (Join-Path $PSScriptRoot "..")
try {
    if (-not (Test-Path "node_modules")) {
        Write-Host "Installing frontend dependencies (first run)..." -ForegroundColor Cyan
        npm install
    }
    $env:VITE_DEMO_MODE = "false"
    npm run dev
}
finally {
    Pop-Location
}
