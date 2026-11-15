"""Does a conversion keep the moment-to-moment movement of the input?

    PYTHONPATH=$PWD .venv/bin/python scripts/dynamics.py INPUT.wav label=OUTPUT.wav ...

Per 10 ms frame of non-silent audio: brightness (2-8 kHz against 250 Hz-2 kHz, dB),
openness (700 Hz-2 kHz against 250-700 Hz, dB) and level. Reports how widely each
moves in the input and in each output, and how well the output follows the input.
"""
import sys, os
import torch
sys.path.insert(0, os.path.dirname(__file__))
from probe import load48

def feats(x):
    S = torch.stft(x, 1024, 480, window=torch.hann_window(1024), return_complex=True).abs().pow(2); f = torch.arange(513) * 48000 / 1024
    e = lambda lo, hi: 10 * torch.log10(S[(f >= lo) & (f < hi)].sum(0) + 1e-10)
    return torch.stack([e(2000, 8000) - e(250, 2000), e(700, 2000) - e(250, 700), e(80, 16000)])

def align(a, b):
    ea, eb = (torch.nn.functional.avg_pool1d(x.abs()[None, None], 480, 480)[0, 0] for x in (a, b)); n = min(len(ea), len(eb))
    k = max(range(-150, 151), key=lambda k: float((ea[max(0, -k):n - max(0, k)] * eb[max(0, k):n - max(0, -k)]).sum())) * 480
    a, b = (a, b[k:]) if k > 0 else (a[-k:], b); n = min(len(a), len(b)); return a[:n], b[:n]

if __name__ == "__main__":
    x = load48(sys.argv[1]); names = ("brightness", "openness", "level")
    print(f"{'':22}" + "".join(f"{n + ' spread':>18}{'follows input':>15}" for n in names))
    for spec in ["input=" + sys.argv[1]] + sys.argv[2:]:
        label, p = spec.split("=", 1); a, b = align(x, load48(p)); A, B = feats(a), feats(b); T = min(A.shape[1], B.shape[1]); A, B = A[:, :T], B[:, :T]
        act = A[2] > A[2].max() - 35; row = ""
        for i in range(3):
            u, v = A[i, act], B[i, act]
            row += f"{float(v.quantile(.9) - v.quantile(.1)):15.1f} dB{float(torch.corrcoef(torch.stack([u, v]))[0, 1]):15.2f}"
        print(f"{label:22}{row}")
