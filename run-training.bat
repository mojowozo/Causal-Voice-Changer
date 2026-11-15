@echo off
rem Sets up everything on first run (Python packages, pretrained files), then opens the training window.
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
echo Checking pretrained files...
uv run --no-sync python -m causalvc.weights
if errorlevel 1 goto failed
uv run --no-sync python -m causalvc.gui
if errorlevel 1 goto failed
exit /b 0

:failed
echo.
echo Setup did not finish. The message above says what went wrong.
pause
exit /b 1
