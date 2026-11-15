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

rem The Windows GPU build of PyTorch: "cuda" for NVIDIA, "rocm" for AMD (see scripts\gpu-extra.ps1).
if not defined CAUSALVC_GPU (
    for /f "usebackq delims=" %%g in (`powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\gpu-extra.ps1"`) do set "CAUSALVC_GPU=%%g"
)
if not defined CAUSALVC_GPU set "CAUSALVC_GPU=cuda"

echo Checking Python packages (PyTorch for %CAUSALVC_GPU%)...
uv sync --extra %CAUSALVC_GPU%
if errorlevel 1 goto failed

rem The Windows front end (realtime\winui, WinUI 3) is built once with the .NET SDK. Without
rem the SDK, the small window in causalvc\live_gui.py opens instead.
rem Set CAUSALVC_REBUILD=1 to build it again (after updating the project).
set "APP=realtime\winui\out\CausalVC.Realtime.exe"
if defined CAUSALVC_REBUILD if exist "%APP%" del "%APP%"
if not exist "%APP%" (
    where dotnet >nul 2>nul
    if errorlevel 1 goto small_window
    echo Building the voice changer window ^(first run only^)...
    dotnet publish "realtime\winui\CausalVC.Realtime.csproj" -c Release -o "realtime\winui\out"
    if errorlevel 1 goto failed
)
set "CAUSALVC_REALTIME_DIR=%~dp0realtime"
start "" "%APP%"
exit /b 0

:small_window
echo The .NET SDK was not found, so the small window opens instead of the full app.
echo For the full app, install the .NET 10 SDK ^(winget install Microsoft.DotNet.SDK.10^) and run this again.
uv run --no-sync python -m causalvc.live_gui
if errorlevel 1 goto failed
exit /b 0

:failed
echo.
echo Something went wrong.
pause
exit /b 1
