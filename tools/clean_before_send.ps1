# 打包前清理：删掉日志、字节码缓存与临时渲染目录
# 用法：
#   powershell -ExecutionPolicy Bypass -File tools\clean_before_send.ps1
#   powershell -ExecutionPolicy Bypass -File tools\clean_before_send.ps1 -RemoveEnv   # 连 .env 一起删（会丢 Key）
param(
    [switch]$RemoveEnv,
    [switch]$WhatIfOnly
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$removed = @()

# 1) 日志目录
if (Test-Path (Join-Path $root 'logs')) {
    if (-not $WhatIfOnly) { Remove-Item (Join-Path $root 'logs') -Recurse -Force }
    $removed += 'logs/'
}

# 2) __pycache__ 与 *.pyc
Get-ChildItem $root -Recurse -Directory -Filter '__pycache__' -ErrorAction SilentlyContinue | ForEach-Object {
    if (-not $WhatIfOnly) { Remove-Item $_.FullName -Recurse -Force }
    $removed += $_.FullName.Substring($root.Length + 1)
}
Get-ChildItem $root -Recurse -File -Filter '*.pyc' -ErrorAction SilentlyContinue | ForEach-Object {
    if (-not $WhatIfOnly) { Remove-Item $_.FullName -Force }
    $removed += $_.FullName.Substring($root.Length + 1)
}

# 3) 渲染/预览临时目录
Get-ChildItem $root -Directory -Filter '.dsh-office-*' -ErrorAction SilentlyContinue | ForEach-Object {
    if (-not $WhatIfOnly) { Remove-Item $_.FullName -Recurse -Force }
    $removed += $_.Name
}

# 4) 可选：删 .env（含 Key）
if ($RemoveEnv -and (Test-Path (Join-Path $root '.env'))) {
    if (-not $WhatIfOnly) { Remove-Item (Join-Path $root '.env') -Force }
    $removed += '.env'
}

if ($removed.Count -eq 0) {
    "已是最干净状态，无需清理。"
} else {
    "已清理 $($removed.Count) 项：" + (($removed | Select-Object -Unique) -join '、')
}

# 5) 剩余敏感文件提示
$left = @()
if (Test-Path (Join-Path $root '.env')) { $left += '.env（含 API Key）' }
Get-ChildItem (Join-Path $root 'outputs\metrics') -Filter 'qa_scoring_filled*.xlsx' -ErrorAction SilentlyContinue | ForEach-Object {
    $left += $_.Name + '（个人评分）'
}
if ($left.Count -gt 0) {
    Write-Host "`n注意，以下文件仍属于个人数据，打包前请确认：" -ForegroundColor Yellow
    $left | ForEach-Object { "  - $_" }
} else {
    "`n未发现个人数据残留。"
}
