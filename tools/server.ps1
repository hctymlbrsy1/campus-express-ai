# 后台服务管理：start / stop / status
# 用法：powershell -ExecutionPolicy Bypass -File tools\server.ps1 start
param(
    [ValidateSet('start', 'stop', 'status')]
    [string]$Action = 'status',
    [int]$Port = 7860
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$pidFile = Join-Path $root 'logs\server.pid'
$logDir = Split-Path $pidFile
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }

# Python 解析：环境变量 EXPRESS_PYTHON → .venv → PATH 的 python → py 启动器
# 注意：必须实际执行一次 -c，跳过 Windows 应用商店的 python.exe 占位符（它会直接退出）。
$py = $null
$candidates = @(
    $env:EXPRESS_PYTHON,                       # 当前进程
    [Environment]::GetEnvironmentVariable('EXPRESS_PYTHON', 'User'),  # 已持久化但本进程未继承
    (Join-Path $root '.venv\Scripts\python.exe'),
    (Join-Path $root '.venv\bin\python'),
    'python',
    'py'
)
foreach ($cand in $candidates) {
    if (-not $cand) { continue }
    $cmd = Get-Command $cand -ErrorAction SilentlyContinue
    if (-not $cmd) { continue }
    $exe = $cmd.Source
    try {
        $probe = & $exe -c "import sys;print(sys.version_info[0])" 2>$null
        if ($LASTEXITCODE -eq 0 -and "$probe".Trim() -eq '3') { $py = $exe; break }
    } catch { }
}
if (-not $py) { throw '找不到可用的 Python 3，请安装 Python 3.10+ 或创建 .venv' }

function Get-ServerPid {
    if (Test-Path $pidFile) {
        $p = Get-Content $pidFile -ErrorAction SilentlyContinue
        if ($p -and (Get-Process -Id $p -ErrorAction SilentlyContinue)) { return [int]$p }
    }
    return $null
}

function Test-Health {
    try {
        $r = Invoke-RestMethod "http://127.0.0.1:$Port/api/health" -TimeoutSec 5
        return ($r | ConvertTo-Json -Compress)
    } catch { return $null }
}

switch ($Action) {
    'start' {
        if (Get-ServerPid) { "已在运行，PID $(Get-ServerPid)"; break }
        $proc = Start-Process -FilePath $py -ArgumentList 'app.py', '--port', $Port `
            -WorkingDirectory (Join-Path $root 'src') -WindowStyle Hidden -PassThru
        $proc.Id | Set-Content $pidFile -Encoding ASCII
        # 轮询最多 20 秒：Flask 启动时导入 pandas/sklearn 可能超过 4 秒，固定 sleep 会误报失败
        $h = $null
        for ($i = 0; $i -lt 40; $i++) {
            Start-Sleep -Milliseconds 500
            $h = Test-Health
            if ($h) { break }
            if ($proc.HasExited) { break }
        }
        if ($h) { "已启动，PID $($proc.Id)`n$h" } else { "启动失败，请看 logs\app.log" }
    }
    'stop' {
        $p = Get-ServerPid
        if (-not $p) { "没有在运行（或 pid 文件已失效）"; break }
        Stop-Process -Id $p -Force
        Remove-Item $pidFile -ErrorAction SilentlyContinue
        "已停止 PID $p"
    }
    'status' {
        $p = Get-ServerPid
        $h = Test-Health
        if ($h) { "运行中，PID $p`n$h" }
        else { "无响应" + $(if ($p) { "（PID $p 存在但服务不通）" } else { "（未启动）" }) }
    }
}
