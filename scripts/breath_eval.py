"""How much voice against breath is in a recording, compared with a reference take of the same performance?

    PYTHONPATH=$PWD .venv/bin/python scripts/breath_eval.py DRY.wav label=OTHER.wav ...

On the frames where the dry take is voiced: the share the other file is also voiced, and in each
band the energy on the harmonics against the energy between them (dB; higher = more voice, less breath).
Pitch is read from each file separately, so a pitch-shifted conversion can be compared too, but
compare like with like where possible (same pitch) because the measure depends a little on pitch.
"""
import os, sys
import torch
sys.path.insert(0, os.path.dirname(__file__))
from probe import DEV, load48, rmvpe_f0
from dynamics import align

BANDS = ((300, 1000), (1000, 2000), (2000, 4000), (4000, 8000))

def hnr(x, f0, mask):
    S = torch.stft(x, 2048, 480, window=torch.hann_window(2048), return_complex=True).abs().pow(2); fr = torch.arange(1025) * 48000 / 2048
    T = min(S.shape[1], len(f0), len(mask)); ts = torch.nonzero(mask[:T] & (f0[:T] > 0))[:, 0]
    if len(ts) < 5: return None, 0
    ph = (fr[None] / f0[ts][:, None]) % 1.0; Sp = S[:, ts].T; out = []
    for lo, hi in BANDS:
        b = ((fr >= lo) & (fr < hi))[None]; on = (b & ((ph < 0.2) | (ph > 0.8))).float(); off = (b & (ph > 0.35) & (ph < 0.65)).float()
        r = 10 * torch.log10(((Sp * on).sum(1) / on.sum(1).clamp_min(1)) / ((Sp * off).sum(1) / off.sum(1).clamp_min(1)).clamp_min(1e-12) + 1e-12)
        out.append(float(r.median()))
    return out, len(ts)

if __name__ == "__main__":
    dry = load48(sys.argv[1])
    print(f"{'':30}{'also voiced':>12}" + "".join(f"{f'{lo}-{hi} Hz':>14}" for lo, hi in BANDS) + f"{'frames':>8}")
    for spec in ["dry=" + sys.argv[1]] + sys.argv[2:]:
        label, p = spec.split("=", 1); a, b = align(dry, load48(p)); fa, fb = rmvpe_f0(a).cpu(), rmvpe_f0(b).cpu(); T = min(len(fa), len(fb)); va = fa[:T] > 0
        h, n = hnr(b, fb[:T], va)
        print(f"{label:30}{float((fb[:T][va] > 0).float().mean()):12.0%}" + ("".join(f"{v:14.1f}" for v in h) if h else "") + f"{n:8d}")
