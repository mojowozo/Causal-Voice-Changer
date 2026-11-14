@echo off
rem Opens the live voice changer window: microphone in, converted voice out.
setlocal
cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 (
    echo uv ^(the Python package manager this project uses^) was not found. Installing it from https://astral.sh/uv ...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)

echo Checking Python packages...
uv sync
if errorlevel 1 goto failed
uv run --no-sync python -m causalvc.live_gui
if errorlevel 1 goto failed
exit /b 0

:failed
echo.
echo Something went wrong. The message above says what.
pause
exit /b 1
