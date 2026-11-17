# A fingerprint of a Windows front end's source files (a WinUI project folder, plus nuget.config).
# A pre-built window from a release download carries the fingerprint of the sources it was built
# from (out\source-hash.txt), so run-realtime.bat and run-training.bat start it as is while the
# sources still match, instead of comparing file times (which an unpacked download doesn't keep in
# a useful order). Line endings don't count: git may check files out with CRLF or LF.
#
#   ui-source-hash.ps1 -Folder realtime\winui                 prints the fingerprint
#   ui-source-hash.ps1 -Folder realtime\winui -Write FILE     saves it to FILE
#   ui-source-hash.ps1 -Folder realtime\winui -Check FILE     exits 0 when FILE holds it, else 1
#
# Run from the project folder (paths in the fingerprint are relative to it).
param(
    [Parameter(Mandatory = $true)][string]$Folder,
    [string]$Write,
    [string]$Check
)
$ErrorActionPreference = 'Stop'

$root = (Get-Location).Path.TrimEnd('\') + '\'
$files = @(Get-ChildItem $Folder -Recurse -File | Where-Object {
        $_.FullName -notmatch '\\(out|bin|obj|\.vs)\\' -and $_.Name -notlike '*.user'
    })
if (Test-Path 'nuget.config') { $files += Get-Item 'nuget.config' }

$sha = [Security.Cryptography.SHA256]::Create()
function Hex([byte[]]$bytes) { ([BitConverter]::ToString($sha.ComputeHash($bytes)) -replace '-', '').ToLowerInvariant() }
$binary = @('.ico', '.png', '.jpg', '.gif', '.bmp')

$lines = foreach ($file in $files) {
    $relative = $file.FullName.Substring($root.Length).Replace('\', '/')
    if ($binary -contains $file.Extension.ToLowerInvariant()) {
        $bytes = [IO.File]::ReadAllBytes($file.FullName)
    } else {
        $bytes = [Text.Encoding]::UTF8.GetBytes([IO.File]::ReadAllText($file.FullName).Replace("`r", ''))
    }
    "$(Hex $bytes) $relative"
}
$lines = [string[]]$lines
[Array]::Sort($lines, [StringComparer]::Ordinal)
$hash = Hex ([Text.Encoding]::UTF8.GetBytes(($lines -join "`n") + "`n"))

# (.NET resolves relative paths against the process's folder, not PowerShell's location.)
function FullPath([string]$path) { $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($path) }
if ($Check) {
    if ((Test-Path $Check) -and ([IO.File]::ReadAllText((FullPath $Check)).Trim() -eq $hash)) { exit 0 }
    exit 1
}
if ($Write) {
    [IO.File]::WriteAllText((FullPath $Write), $hash + "`n")
} else {
    $hash
}
