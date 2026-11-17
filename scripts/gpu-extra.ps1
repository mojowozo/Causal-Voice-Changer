# Prints which Windows build of PyTorch to install: "cuda" (NVIDIA), "rocm" (AMD, newer Radeons) or
# "rocm-rdna2" (AMD's nightly build, for the Radeon RX 6000 series).
# Used by run-training.bat and run-realtime.bat; set CAUSALVC_GPU=cuda, rocm or
# rocm-rdna2 to choose yourself.
#
# NVIDIA wins when both are present (a laptop with an AMD processor's graphics and an NVIDIA card).
# AMD's ROCm build supports Radeon RX 7000 and 9000, Radeon PRO W7000 and Ryzen AI Max / AI 300
# graphics. It cannot start on the RX 6000 series (an RX 6700 XT, say), whatever the driver: those get
# AMD's newer nightly build (TheRock), which supports them. Older Radeons get the standard build,
# which runs voices on the processor: slower, so use a chunk of 60 ms or more.
$names = @(Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue | ForEach-Object { $_.Name })
# Shown in the launcher's window (stderr: only stdout is the answer), so a wrong pick is easy to spot.
[Console]::Error.WriteLine("Graphics found: $(if ($names) { $names -join ', ' } else { 'none reported' })")
if ($names -match 'NVIDIA') {
    'cuda'
} elseif ($names -match 'AMD|Radeon') {
    $supported = 'RX 7\d{3}|RX 9\d{3}|PRO W7\d{3}|Radeon 8\d{2}0S|Radeon 8\d0M'
    if ($names -match $supported) {
        'rocm'
    } elseif ($names -match 'RX 6\d{3}|PRO W6\d{3}') {
        [Console]::Error.WriteLine("Radeon RX 6000 series: using AMD's newer nightly build of ROCm, which supports it.")
        'rocm-rdna2'
    } else {
        [Console]::Error.WriteLine("AMD's GPU runtime (ROCm) does not support this graphics card, so voices run on the processor. Set Chunk to 60 ms or more on the Performance page.")
        'cuda'
    }
} else {
    'cuda'
}
