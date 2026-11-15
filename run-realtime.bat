@echo off
rem Opens the live voice changer window: microphone in, converted voice out.
setlocal
cd /d "%~dp0"
rem A Windows environment of its own: the default .venv may belong to Linux when this folder is shared between systems.
set "UV_PROJECT_ENVIRONMENT=.venv-win"

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
echo Something went wrong.
pause
exit /b 1
