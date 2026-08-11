# Start deepseek-cursor-proxy with Cloudflare Tunnel (no ngrok).
# Usage: .\scripts\start-with-cloudflare.ps1 [-Port 9000]

param(
    [int]$Port = 9000
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

$env:Path = "C:\Users\Lenovo\.local\bin;" +
    [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
    [System.Environment]::GetEnvironmentVariable("Path", "User")

if (-not (Get-Command cloudflared -ErrorAction SilentlyContinue)) {
    Write-Error "cloudflared not found. Install with: winget install Cloudflare.cloudflared"
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "uv not found. Install from https://docs.astral.sh/uv/"
}

Write-Host "Starting proxy on http://127.0.0.1:$Port (ngrok disabled)..." -ForegroundColor Cyan
$proxy = Start-Process -FilePath "uv" `
    -ArgumentList @("run", "deepseek-cursor-proxy", "--no-ngrok", "--port", $Port) `
    -WorkingDirectory $ProjectRoot `
    -PassThru `
    -NoNewWindow

Start-Sleep -Seconds 2

Write-Host "Starting Cloudflare quick tunnel..." -ForegroundColor Cyan
Write-Host "Cursor Base URL will be: https://<random>.trycloudflare.com/v1" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop both proxy and tunnel." -ForegroundColor Gray
Write-Host ""

try {
    cloudflared tunnel --url "http://127.0.0.1:$Port"
}
finally {
    if (-not $proxy.HasExited) {
        Write-Host "`nStopping proxy..." -ForegroundColor Gray
        Stop-Process -Id $proxy.Id -Force -ErrorAction SilentlyContinue
    }
}
