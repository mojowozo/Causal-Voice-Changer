@echo off
rem Sets up everything on first run (Python packages, pretrained files), then opens the training window.
setlocal
cd /d "%~dp0"
rem Python packages (with PyTorch for this graphics card) and pretrained files, shared with the
rem voice changer (run-realtime.bat does the same). Also sets UV_PROJECT_ENVIRONMENT.
call scripts\setup.bat training
if errorlevel 1 goto failed

rem The Windows front end (training\winui, WinUI 3) is built once with the .NET 10 SDK, installed
rem into .dotnet in this folder when missing (no admin rights). If that fails, the small window in
rem causalvc\gui.py opens instead. Set CAUSALVC_REBUILD=1 to build it again (after updating).
set "APP=training\winui\out\CausalVC.Training.exe"
set "STAMP=training\winui\out\source-hash.txt"
if defined CAUSALVC_REBUILD (
    if exist "%APP%" del "%APP%"
    if exist "%STAMP%" del "%STAMP%"
)
rem A window that came built in a release download holds a fingerprint of the sources it was built
rem from (file times don't survive unpacking in a useful order): start it while those still match.
if exist "%APP%" if exist "%STAMP%" (
    powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\ui-source-hash.ps1" -Folder "training\winui" -Check "%STAMP%"
    if errorlevel 1 (
        echo The training window was updated; building it again...
        del "%APP%" "%STAMP%"
    ) else (
        goto launch
    )
)
rem After an update (git pull): rebuild when any of the window's source files is newer than the program.
if exist "%APP%" (
    powershell -NoProfile -Command "$exe = (Get-Item '%APP%').LastWriteTime; $new = Get-ChildItem 'training\winui', 'nuget.config' -Recurse -File -Include *.cs,*.xaml,*.csproj,*.config | Where-Object { $_.FullName -notmatch '\\(out|bin|obj)\\' -and $_.LastWriteTime -gt $exe }; if ($new) { exit 1 }"
    if errorlevel 1 (
        echo The training window was updated; building it again...
        del "%APP%"
    )
)
if exist "%APP%" goto launch

call :find_dotnet
if not defined DOTNET goto small_window
echo Building the training window ^(first run only^)...
"%DOTNET%" publish "training\winui\CausalVC.Training.csproj" -c Release -o "training\winui\out"
if errorlevel 1 goto small_window_after_build
if not exist "%APP%" goto small_window_after_build

:launch
set "CAUSALVC_DIR=%~dp0."
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

:small_window_after_build
echo The training window could not be built; the small window opens instead.
goto open_small_window

:small_window
echo The .NET 10 SDK could not be installed, so the small window opens instead of the full app.
echo For the full app, install it ^(winget install Microsoft.DotNet.SDK.10^) and run this again.
:open_small_window
uv run --no-sync python -m causalvc.gui
if errorlevel 1 goto failed
exit /b 0

:failed
echo.
echo Setup did not finish. The message above says what went wrong.
pause
exit /b 1
