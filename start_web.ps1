<#
.SYNOPSIS
  One-click launcher for the Influence Strategy Web Console (backend + frontend).

.DESCRIPTION
  - Auto-locates project root (the script's own folder)
  - Starts the backend with the project venv (.venv\Scripts\python.exe)
  - Runs pip install -e . on first launch (so uvicorn can import influence_strategy)
  - Runs npm install + npm run build only when dist/ is missing or stale
  - prod mode: uvicorn in the current window, Ctrl-C to stop
  - dev  mode: also opens a second window running vite (hot reload) on 5173

.PARAMETER Mode
  prod: backend serves the built frontend, open http://127.0.0.1:<Port>
  dev  : also start vite dev server, open http://127.0.0.1:5173

.PARAMETER PythonExe
  Path of the Python interpreter. Default resolution order:
  <project>\.venv\Scripts\python.exe -> D:\python\python.exe -> python (PATH)

.PARAMETER Port
  uvicorn port, default 8000

.EXAMPLE
  .\start_web.ps1
  .\start_web.ps1 -Mode dev
  .\start_web.ps1 -Port 9000
#>

[CmdletBinding()]
param(
    [ValidateSet('prod', 'dev')]
    [string]$Mode = 'prod',
    [string]$PythonExe = '',
    [int]$Port = 8000
)

$ErrorActionPreference = 'Stop'

# 1. Project root = this script's folder
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path $ScriptDir
Set-Location $ProjectRoot
Write-Host "[1/5] Project root: $ProjectRoot" -ForegroundColor Cyan

# 2. Check the Python interpreter
# 依次尝试：显式传入 → 项目 .venv → 已装好依赖的系统 Python → PATH 里的 python。
#
# 只认 .venv 这一个默认值会让没建过 venv 的机器必然启动失败：本项目目前就没有
# .venv，依赖装在 D:\python 下。所以改成逐级回退 —— 有 venv 的机器照旧用 venv，
# 没有的机器自动退到系统 Python，两边都能一键启动。
if (-not $PythonExe) {
    $candidates = @(
        (Join-Path $ProjectRoot '.venv\Scripts\python.exe'),
        'D:\python\python.exe',
        'python'
    )
    foreach ($candidate in $candidates) {
        if (Get-Command $candidate -ErrorAction SilentlyContinue) {
            $PythonExe = $candidate
            break
        }
    }
}
if (-not $PythonExe) {
    throw "Python interpreter not found (pass -PythonExe <path>)"
}
$pyVersion = & $PythonExe --version 2>&1
Write-Host "[2/5] Python: $pyVersion" -ForegroundColor Cyan

# 3. Make influence_strategy importable for uvicorn (one-time)
$checkImport = & $PythonExe -c "import influence_strategy" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[3/5] Running: pip install -e ." -ForegroundColor Yellow
    & $PythonExe -m pip install -e . | Select-Object -Last 3
} else {
    Write-Host "[3/5] influence_strategy is importable" -ForegroundColor Cyan
}

# 4. Frontend deps and build
$feDir = Join-Path $ProjectRoot 'web\frontend'
$distDir = Join-Path $feDir 'dist'
$nodeModules = Join-Path $feDir 'node_modules'

$npmCache = Join-Path $feDir '.npm-cache'
if (-not (Test-Path $nodeModules)) {
    Write-Host "[4/5] npm install (first run)" -ForegroundColor Yellow
    Push-Location $feDir
    try {
        npm install --cache $npmCache --no-audit --no-fund | Select-Object -Last 5
        if ($LASTEXITCODE -ne 0) { throw "npm install failed (exit $LASTEXITCODE). Delete node_modules and retry." }
    } finally { Pop-Location }
} else {
    Write-Host "[4/5] node_modules ready" -ForegroundColor Cyan
}

$needBuild = $true
if (Test-Path $distDir) {
    $srcFiles = Get-ChildItem (Join-Path $feDir 'src') -Recurse -File -ErrorAction SilentlyContinue
    $distMTime = (Get-Item $distDir).LastWriteTime
    $newer = $srcFiles | Where-Object { $_.LastWriteTime -gt $distMTime } | Select-Object -First 1
    if (-not $newer) { $needBuild = $false }
}
if ($needBuild) {
    Write-Host "      Building frontend -> dist" -ForegroundColor Yellow
    Push-Location $feDir
    try {
        npm run build | Select-Object -Last 12
        if ($LASTEXITCODE -ne 0) { throw "npm run build failed (exit $LASTEXITCODE)." }
    } finally { Pop-Location }
} else {
    Write-Host "      dist is up to date, skip build" -ForegroundColor Cyan
}

if (-not (Test-Path (Join-Path $distDir 'index.html'))) {
    throw "dist/index.html not found - frontend build did not produce output."
}

# 5. Start the backend
$env:PYTHONPATH = Join-Path $ProjectRoot 'src'
$apiUrl = "http://127.0.0.1:$Port"
$webUrl = if ($Mode -eq 'dev') { 'http://127.0.0.1:5173' } else { $apiUrl }

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
if ($Mode -eq 'dev') {
    Write-Host "  Mode : dev (uvicorn + vite hot reload)" -ForegroundColor Green
    Write-Host "  Open : $webUrl (vite proxies /api -> $apiUrl)" -ForegroundColor Green
    Write-Host "  Port $Port serves the API / built frontend directly" -ForegroundColor Green
} else {
    Write-Host "  Mode : prod (uvicorn serves API + frontend)" -ForegroundColor Green
    Write-Host "  Open : $webUrl" -ForegroundColor Green
}
Write-Host "  Stop : Ctrl-C" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

if ($Mode -eq 'dev') {
    $uvicornArgs = @('-m', 'uvicorn', 'influence_strategy.web_app:app',
                     '--host', '127.0.0.1', '--port', "$Port")
    $uvicornProc = Start-Process -FilePath $PythonExe `
        -ArgumentList $uvicornArgs `
        -WorkingDirectory $ProjectRoot `
        -PassThru `
        -RedirectStandardOutput (Join-Path $ProjectRoot 'uvicorn.out.log') `
        -RedirectStandardError  (Join-Path $ProjectRoot 'uvicorn.err.log') `
        -WindowStyle Hidden

    $viteCmd = "cd `"$feDir`"; npm run dev"
    Start-Process -FilePath "powershell.exe" `
        -ArgumentList @('-NoExit', '-Command', $viteCmd) `
        -WindowStyle Normal | Out-Null

    Write-Host "uvicorn PID: $($uvicornProc.Id) | logs: uvicorn.out.log / uvicorn.err.log" -ForegroundColor Yellow
    Write-Host "Waiting for vite to start (5-10s), then open $webUrl" -ForegroundColor Yellow
    Write-Host "Press Ctrl-C to stop the backend (close the vite window manually)" -ForegroundColor Yellow

    try {
        while ($true) { Start-Sleep -Seconds 3600 }
    } finally {
        if (-not $uvicornProc.HasExited) {
            Stop-Process -Id $uvicornProc.Id -Force -ErrorAction SilentlyContinue
        }
    }
} else {
    & $PythonExe -m uvicorn influence_strategy.web_app:app `
        --host 127.0.0.1 --port $Port
}
