@echo off
chcp 65001 >nul
cd /d "%~dp0"
if /i "%~1"=="stop" (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\launch.ps1" -Stop
) else (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\launch.ps1"
)
if errorlevel 1 pause
