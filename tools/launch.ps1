# 一键启动器（试验版）：检查环境 → 需要时补依赖/重训 → 起服务 → 打开浏览器
# 由 start.bat 调用，也可以直接运行：
#   powershell -ExecutionPolicy Bypass -File tools\launch.ps1
#   powershell -ExecutionPolicy Bypass -File tools\launch.ps1 -Stop
#   powershell -ExecutionPolicy Bypass -File tools\launch.ps1 -Port 7861 -NoBrowser
param(
    [int]$Port = 7860,
    [switch]$Stop,
    [switch]$NoBrowser,
    [switch]$ForceRetrain
)

$ErrorActionPreference = 'Stop'
try { [Console]::OutputEncoding = [Text.Encoding]::UTF8 } catch { }

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Info($msg) { Write-Host "  $msg" }
function Step($msg) { Write-Host "`n>> $msg" -ForegroundColor Cyan }
function Fail($msg) { Write-Host "`n[!] $msg" -ForegroundColor Red }

# 停止
if ($Stop) {
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $root 'tools\server.ps1') stop
    exit 0
}

Write-Host "===== 校园快递驿站智能服务系统 · 启动器 =====" -ForegroundColor Green
Info "项目目录：$root"

# 1) 找 Python
Step "检查 Python"
$py = $null
$cands = @(
    $env:EXPRESS_PYTHON,
    [Environment]::GetEnvironmentVariable('EXPRESS_PYTHON', 'User'),
    (Join-Path $root '.venv\Scripts\python.exe'),
    (Join-Path $root '.venv\bin\python'),
    'python',
    'py'
)
foreach ($c in $cands) {
    if (-not $c) { continue }
    $cmd = Get-Command $c -ErrorAction SilentlyContinue
    if (-not $cmd) { continue }
    try {
        $probe = & $cmd.Source -c "import sys;print(sys.version_info[0])" 2>$null
        if ($LASTEXITCODE -eq 0 -and "$probe".Trim() -eq '3') { $py = $cmd.Source; break }
    } catch { }
}
if (-not $py) {
    Fail "没找到可用的 Python 3。请安装 Python 3.10+（安装时勾选 Add to PATH），或先创建 .venv。"
    exit 1
}
Info "使用：$py"
Info (& $py -V 2>&1)

# 2) 检查依赖，缺了就装
Step "检查依赖"
& $py -c "import flask, sklearn, pandas, joblib, numpy" 2>$null
if ($LASTEXITCODE -ne 0) {
    Info "依赖不齐，开始安装（首次约 1–2 分钟）..."
    & $py -m pip install -r (Join-Path $root 'requirements.txt')
    if ($LASTEXITCODE -ne 0) {
        Fail "依赖安装失败。可尝试换镜像：`$env:PIP_INDEX_URL='https://pypi.tuna.tsinghua.edu.cn/simple' 后重试。"
        exit 1
    }
} else {
    Info "依赖已就绪"
}

# 3) 模型自检；换机器或首次运行时自动重训
Step "检查模型"
$needTrain = $ForceRetrain
if (-not $needTrain) {
    Info "自检中（首次或换机器时会慢一点，随后自动重训）..."
    $out = & $py (Join-Path $root 'tools\doctor.py') 2>&1
    if ($LASTEXITCODE -ne 0) {
        $needTrain = $true
        Info "模型加载或推理失败，需要重训"
    } else {
        Info "模型正常"
    }
}

if ($needTrain) {
    Step "用本机环境重训（约 30 秒）"
    $argsRun = @('src/run_all.py')
    if (Test-Path (Join-Path $root 'data\processed\tickets_clean.csv')) { $argsRun += '--skip-gen' }
    & $py @argsRun
    if ($LASTEXITCODE -ne 0) {
        Fail "重训失败，请把上面的输出发回。"
        exit 1
    }
}

# 4) 启动服务
Step "启动服务（端口 $Port）"
$health = $null
try { $health = Invoke-RestMethod "http://127.0.0.1:$Port/api/health" -TimeoutSec 2 } catch { }
if ($health) {
    Info "端口 $Port 已有实例在运行，直接复用"
} else {
    $out = & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $root 'tools\server.ps1') start -Port $Port 2>&1
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Milliseconds 500
        try { $health = Invoke-RestMethod "http://127.0.0.1:$Port/api/health" -TimeoutSec 2; break } catch { }
    }
    if (-not $health) {
        Fail "服务没有起来。日志：$root\logs\app.log"
        $out | Select-Object -Last 5 | ForEach-Object { Info $_ }
        exit 1
    }
}

$url = "http://127.0.0.1:$Port"
Info "服务已就绪：$url"
Info ("模型状态：" + (($health.models.PSObject.Properties | ForEach-Object { "$($_.Name)=$($_.Value)" }) -join ' '))
if ($health.qa_mode) { Info "问答模式：$($health.qa_mode)" }

if (-not $NoBrowser) {
    Start-Process $url
    Info "浏览器已打开"
}

Write-Host "`n完成。关闭本窗口不影响服务。" -ForegroundColor Green
Write-Host "停止服务：双击 start.bat stop，或运行 tools\server.ps1 stop" -ForegroundColor DarkGray
if (-not $NoBrowser) { Start-Sleep -Seconds 3 }
