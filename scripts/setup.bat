@echo off
rem Shared first-run setup for both apps, run by run-realtime.bat and run-training.bat:
rem
rem     call scripts\setup.bat realtime^|training
rem
rem 1. uv (the Python package manager this project uses), installed when missing.
rem 2. The Python packages both apps share (.venv-win), with the PyTorch build for this graphics card;
rem    quick when nothing changed. Then a check that PyTorch loads.
rem 3. The pretrained files the app needs in weights\: the base every voice runs on (both apps), and
rem    RMVPE, which labels training audio (training only).
rem
rem Whichever app is opened first installs what the other needs too, so either can be run first.
rem No setlocal: the caller keeps UV_PROJECT_ENVIRONMENT, CAUSALVC_GPU and PATH as set here.
rem Exits with 1 when something failed (the message above it says what).
if "%~1"=="" (
    echo usage: scripts\setup.bat realtime^|training
    exit /b 1
)
pushd "%~dp0.."
rem A Windows environment of its own: the default .venv may belong to Linux when this folder is shared between systems.
set "UV_PROJECT_ENVIRONMENT=.venv-win"

where uv >nul 2>nul
if errorlevel 1 set "PATH=%USERPROFILE%\.local\bin;%PATH%"
where uv >nul 2>nul
if errorlevel 1 (
    echo uv ^(the Python package manager this project uses^) was not found. Installing it from https://astral.sh/uv ...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
)

rem The Windows GPU build of PyTorch: "cuda" for NVIDIA, "rocm" for AMD (see scripts\gpu-extra.ps1).
rem A CAUSALVC_GPU set beforehand (here or in Windows settings) skips detecting the graphics card.
if defined CAUSALVC_GPU echo Using CAUSALVC_GPU=%CAUSALVC_GPU%, set before this launcher started ^(the graphics card was not checked^).
if not defined CAUSALVC_GPU (
    for /f "usebackq delims=" %%g in (`powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\gpu-extra.ps1"`) do set "CAUSALVC_GPU=%%g"
)
if not defined CAUSALVC_GPU set "CAUSALVC_GPU=cuda"

echo Checking Python packages (PyTorch for %CAUSALVC_GPU%)...
uv sync --extra %CAUSALVC_GPU%
if errorlevel 1 goto failed
rem PyTorch must load: AMD's ROCm build fails to start on an old driver or graphics it doesn't support.
rem Say what to do then, instead of opening a window that cannot start.
uv run --no-sync python -m causalvc.torch_check
if errorlevel 1 goto failed
echo Checking pretrained files...
uv run --no-sync python -m causalvc.weights %1
if errorlevel 1 goto failed
popd
exit /b 0

:failed
popd
exit /b 1
