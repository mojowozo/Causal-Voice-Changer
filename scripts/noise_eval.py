"""How random is a decoder's unvoiced output, and does it carry a frame-rate ripple?

    PYTHONPATH=$PWD .venv/bin/python scripts/noise_eval.py INPUT.wav label=decoder.pt:speaker [...] [--write DIR]

Converts INPUT twice with different random noise in the excitation and reports how
correlated the two outputs are in unvoiced audio (real noise: about 0), plus the
100 Hz envelope ripple. --write saves one streamed render per decoder.
"""
import os, sys
import numpy as np, soundfile as sf, torch
sys.path.insert(0, os.path.dirname(__file__))
from probe import DEV, load48
from causalvc.decoder.model import FRAME
from causalvc.infer import CausalVC, extract


def band(x, lo, hi):
    X = torch.fft.rfft(x); f = torch.fft.rfftfreq(len(x), 1 / 48000); X[(f < lo) | (f >= hi)] = 0
    return torch.fft.irfft(X, len(x))


def ripple(x, c=100):
    S = torch.stft(x, 256, 48, window=torch.hann_window(256), return_complex=True).abs().pow(2); f = torch.arange(129) * 48000 / 256
    e = S[(f >= 1500) & (f < 10000)].sum(0); e = e / e.mean() - 1; n = len(e) // 1000 * 1000
    M = torch.stft(e[:n], 1000, 250, window=torch.hann_window(1000), return_complex=True).abs().pow(2).mean(1)
    return 10 * np.log10(float(M[c - 1:c + 2].max() / torch.cat([M[c - 12:c - 3], M[c + 4:c + 13]]).mean()))


args = sys.argv[1:]; out = None
if "--write" in args:
    out = args[args.index("--write") + 1]; del args[args.index("--write"):args.index("--write") + 2]; os.makedirs(out, exist_ok=True)
x = load48(args[0]); x = x[: len(x) // 960 * 960]
cor = lambda a, b: float((a * b).sum() / (a.norm() * b.norm()))
print(f"input ripple at 100 Hz: {ripple(x):.1f} dB")
for item in args[1:]:
    label, rest = item.split("="); dec, spk = rest.rsplit(":", 1)
    vc = CausalVC(dec, DEV); d = vc.dec; s = torch.tensor([vc.speakers.index(spk)], device=DEV)
    with torch.no_grad():
        c, f, l = extract(vc.enc, vc.rmvpe, x.to(DEV)[None]); h, g = d.features(c, f, l, s)
        a, b = (d.generate(h, g, d.source(f[:, : h.shape[-1]]))[0, 0].cpu() for _ in range(2))
    uv = (f[0] == 0).cpu().repeat_interleave(FRAME)[: len(a)]
    share = [cor(band(a, lo, hi)[uv], band(b, lo, hi)[uv]) for lo, hi in ((100, 3000), (3000, 8000), (8000, 20000))]
    print(f"{label}: same-noise share in unvoiced audio (0.1-3 / 3-8 / 8-20 kHz) {' / '.join(f'{v:.2f}' for v in share)}; ripple {ripple(a):.1f} dB")
    if out:
        sf.write(f"{out}/{label}.wav", (a / a.abs().max() * 0.7).numpy(), 48000, subtype="PCM_16")
