<#
.SYNOPSIS
  One-click launcher for the Influence Strategy Web Console (backend + frontend).

.DESCRIPTION
  - Auto-locates project root (the script's own folder)
  - Starts the backend with the project venv (.venv\Scripts\python.exe)
  - Runs pip install -e .[ocr,semantic] on first launch (installs fastapi/uvicorn/numpy/... and OCR engine)
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

# 2. 解析 Python 解释器
# 依次尝试：显式传入 → 项目 .venv → Windows Python Launcher(py -3) → PATH 的 python
# → 兜底 D:\python。注意：D:\python 是作者机器上的绝对路径，不能放最前，否则其它机器
# 上若恰好存在同名但版本不符的解释器会被误用；优先用 PATH / py 里的 python 更易移植。
# 解析结果统一收敛成「绝对路径」，后续所有调用都用它，避免相对路径随 CWD 漂移。
if (-not $PythonExe) {
    $venvPy = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
    if (Test-Path $venvPy) { $PythonExe = $venvPy }

    if (-not $PythonExe -and (Get-Command py -ErrorAction SilentlyContinue)) {
        $PythonExe = & py -3 -c "import sys; print(sys.executable)" 2>$null
    }
    if (-not $PythonExe -and (Get-Command python -ErrorAction SilentlyContinue)) {
        $PythonExe = & python -c "import sys; print(sys.executable)" 2>$null
    }
    if (-not $PythonExe -and (Test-Path 'D:\python\python.exe')) {
        $PythonExe = 'D:\python\python.exe'
    }
}
if (-not $PythonExe) {
    throw "未找到 Python 解释器（可传 -PythonExe <绝对路径> 显式指定）"
}
$PythonExe = "$PythonExe".Trim()
if (-not (Test-Path $PythonExe)) {
    throw "Python 解释器路径无效：$PythonExe"
}
$pyVersion = & $PythonExe --version 2>&1
Write-Host "[2/5] Python: $pyVersion ($PythonExe)" -ForegroundColor Cyan

# 版本要求 >= 3.11（pyproject.toml requires-python）
& $PythonExe -c "import sys; sys.exit(0 if sys.version_info>=(3,11) else 1)" 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Python 版本过低，需要 >= 3.11；当前：$pyVersion"
}

# 3. Python 依赖安装（仅缺失时）
# 关键修正：之前只测 `import influence_strategy`，而该包本身不依赖 fastapi，
# 即使 fastapi/uvicorn 缺失也能「导入成功」→ 误判为已装好 → uvicorn 启动时才报
# ModuleNotFoundError。这里改为逐一检查全部核心运行依赖，缺任一就重装。
$deps = @('influence_strategy', 'fastapi', 'uvicorn', 'pydantic',
          'numpy', 'pandas', 'rapidfuzz', 'networkx', 'sklearn')
$depsOk = $true
foreach ($m in $deps) {
    & $PythonExe -c "import $m" 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { $depsOk = $false; break }
}
if (-not $depsOk) {
    Write-Host "[3/5] 安装 Python 依赖: pip install -e .[ocr,semantic]" -ForegroundColor Yellow
    & $PythonExe -m pip install -e ".[ocr,semantic]" 2>&1 | Select-Object -Last 6
    if ($LASTEXITCODE -ne 0) {
        throw "pip install 失败，请检查网络或手动执行: $PythonExe -m pip install -e .[ocr,semantic]"
    }
} else {
    Write-Host "[3/5] Python 依赖已就绪" -ForegroundColor Cyan
}

# 4. 前端依赖与构建
# 前置检查：本机必须已安装 Node.js（含 npm）。没装时给明确指引，而不是把 npm 报错甩给用户。
if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    throw "未检测到 Node.js（前端必备）。请先安装 Node.js LTS（自带 npm）：https://nodejs.org/ ，安装后重开终端再运行本脚本。"
}
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    throw "未检测到 npm（前端必备）。请安装 Node.js LTS 后重试。"
}
$nodeVer = & node --version 2>&1
Write-Host "[4/5] Node: $nodeVer" -ForegroundColor Cyan

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

# 5. 启动后端
$env:PYTHONPATH = Join-Path $ProjectRoot 'src'
$apiUrl = "http://127.0.0.1:$Port"
$webUrl = if ($Mode -eq 'dev') { 'http://127.0.0.1:5173' } else { $apiUrl }

# 端口占用预检：避免 uvicorn 启动到一半才因端口被占而失败。
try {
    $sock = New-Object System.Net.Sockets.TcpClient
    $sock.Connect('127.0.0.1', $Port)
    $sock.Close()
    Write-Host "[5/5] 警告：端口 $Port 已被占用，uvicorn 可能启动失败。请换 -Port 或先释放该端口。" -ForegroundColor Red
} catch {
    # 端口空闲，正常
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  ✅ 启动完成" -ForegroundColor Green
if ($Mode -eq 'dev') {
    Write-Host "  Mode : dev (uvicorn + vite 热更新)" -ForegroundColor Green
    Write-Host "  Open : $webUrl （vite 把 /api 代理到 $apiUrl）" -ForegroundColor Green
    Write-Host "  Port $Port 同时直接提供 API / 已构建前端" -ForegroundColor Green
    Write-Host "  Stop : 关闭 vite 窗口 + 本窗口 Ctrl-C" -ForegroundColor Green
} else {
    Write-Host "  Mode : prod (uvicorn 同时提供 API + 前端)" -ForegroundColor Green
    Write-Host "  Open : $webUrl" -ForegroundColor Green
    Write-Host "  Stop : Ctrl-C" -ForegroundColor Green
}
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
