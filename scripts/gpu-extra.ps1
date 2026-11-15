# Prints which Windows build of PyTorch to install: "cuda" (NVIDIA) or "rocm" (AMD).
# Used by run-training.bat and run-realtime.bat; set CAUSALVC_GPU=cuda or rocm to choose yourself.
#
# NVIDIA wins when both are present (a laptop with an AMD processor's graphics and an NVIDIA card).
# AMD's ROCm build supports Radeon RX 7000 and 9000, Radeon PRO W7000 and Ryzen AI Max / AI 300
# graphics; on older Radeons either build runs on the CPU instead.
$names = @(Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue | ForEach-Object { $_.Name })
if ($names -match 'NVIDIA') {
    'cuda'
} elseif ($names -match 'AMD|Radeon') {
    $supported = 'RX 7\d{3}|RX 9\d{3}|PRO W7\d{3}|Radeon 8\d{2}0S|Radeon 8\d0M'
    if (-not ($names -match $supported)) {
        [Console]::Error.WriteLine("Note: AMD's ROCm build may not support this graphics card ($($names -join ', ')); if not, voices run on the processor, slowly.")
    }
    'rocm'
} else {
    'cuda'
}
