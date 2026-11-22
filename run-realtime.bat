@echo off
rem Opens the live voice changer window: microphone in, converted voice out.
setlocal
cd /d "%~dp0"
rem Python packages (with PyTorch for this graphics card) and pretrained files, shared with the
rem training app (run-training.bat does the same). Also sets UV_PROJECT_ENVIRONMENT.
call scripts\setup.bat realtime
if errorlevel 1 goto failed

rem The Windows front end (realtime\winui, WinUI 3) is built once with the .NET 10 SDK. When no
rem .NET 10 SDK is installed, one is installed into .dotnet in this folder (no admin rights; delete
rem the folder to remove it). If that fails, the small window in causalvc\live_gui.py opens instead.
rem Set CAUSALVC_REBUILD=1 to build it again (after updating the project).
set "APP=realtime\winui\out\CausalVC.Realtime.exe"
set "STAMP=realtime\winui\out\source-hash.txt"
if defined CAUSALVC_REBUILD (
    if exist "%APP%" del "%APP%"
    if exist "%STAMP%" del "%STAMP%"
)
rem A window that came built in a release download holds a fingerprint of the sources it was built
rem from (file times don't survive unpacking in a useful order): start it while those still match.
if exist "%APP%" if exist "%STAMP%" (
    powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\ui-source-hash.ps1" -Folder "realtime\winui" -Check "%STAMP%"
    if errorlevel 1 (
        echo The voice changer window was updated; building it again...
        del "%APP%" "%STAMP%"
    ) else (
        goto launch
    )
)
rem After an update (git pull): rebuild when any of the window's source files is newer than the program.
if exist "%APP%" (
    powershell -NoProfile -Command "$exe = (Get-Item '%APP%').LastWriteTime; $new = Get-ChildItem 'realtime\winui', 'nuget.config' -Recurse -File -Include *.cs,*.xaml,*.csproj,*.config | Where-Object { $_.FullName -notmatch '\\(out|bin|obj)\\' -and $_.LastWriteTime -gt $exe }; if ($new) { exit 1 }"
    if errorlevel 1 (
        echo The voice changer window was updated; building it again...
        del "%APP%"
    )
)
if exist "%APP%" goto launch

call :find_dotnet
if not defined DOTNET goto small_window
echo Building the voice changer window ^(first run only^)...
"%DOTNET%" publish "realtime\winui\CausalVC.Realtime.csproj" -c Release -o "realtime\winui\out"
if errorlevel 1 goto failed
if not exist "%APP%" (
    echo The build finished but %APP% was not created.
    goto failed
)

:launch
set "CAUSALVC_REALTIME_DIR=%~dp0realtime"
rem The engine runs in the environment checked above (not another one it might find first).
set "CAUSALVC_PYTHON=%~dp0.venv-win\Scripts\python.exe"
start "" "%APP%"
exit /b 0

:find_dotnet
rem Sets DOTNET to a dotnet that has a .NET 10 (or newer) SDK: the one in .dotnet, then the
rem system one, else installs one into .dotnet. An older SDK alone cannot build the project.
set "DOTNET_NOLOGO=1"
set "DOTNET_CLI_TELEMETRY_OPTOUT=1"
set "DOTNET="
set "LOCAL_DOTNET=%~dp0.dotnet\dotnet.exe"
rem (Each dotnet is checked to exist first: piping from a missing program ends the whole script.)
if exist "%LOCAL_DOTNET%" call :has_sdk "%LOCAL_DOTNET%" && (set "DOTNET=%LOCAL_DOTNET%" & exit /b 0)
where dotnet >nul 2>nul && call :has_sdk dotnet && (set "DOTNET=dotnet" & exit /b 0)
echo The .NET 10 SDK was not found. Installing it into .dotnet in this folder
echo ^(about 250 MB to download, first run only^)...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'; $s = Join-Path $env:TEMP 'dotnet-install.ps1'; Invoke-WebRequest 'https://dot.net/v1/dotnet-install.ps1' -OutFile $s -UseBasicParsing; & $s -Channel 10.0 -InstallDir '%~dp0.dotnet' -NoPath"
if exist "%LOCAL_DOTNET%" call :has_sdk "%LOCAL_DOTNET%" && set "DOTNET=%LOCAL_DOTNET%"
exit /b 0

:has_sdk
rem Succeeds when the dotnet given as %1 lists an SDK of version 10 or newer.
"%~1" --list-sdks 2>nul | findstr /r /b /c:"[1-9][0-9]\." >nul
exit /b

:small_window
echo The .NET 10 SDK could not be installed, so the small window opens instead of the full app.
echo For the full app, install it ^(winget install Microsoft.DotNet.SDK.10^) and run this again.
uv run --no-sync python -m causalvc.live_gui
if errorlevel 1 goto failed
exit /b 0

:failed
echo.
echo Something went wrong.
pause
exit /b 1
