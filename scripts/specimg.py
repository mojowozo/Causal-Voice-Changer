"""Side-by-side spectrograms as a grayscale PNG (no plotting library needed).

    python scripts/specimg.py T0 T1 OUT.png name=file.wav:delay_seconds [...]

Top row: 0-6 kHz with fine frequency detail. Bottom row: 0-20 kHz with fine time detail.
"""
import struct, sys, zlib, os
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from probe import load48

t0, t1 = float(sys.argv[1]), float(sys.argv[2]); rows = []
for nfft, fmax in ((2048, 6000), (256, 20000)):
    cols = []
    for item in sys.argv[4:]:
        p, d = item.split("=")[1].rsplit(":", 1); y = load48(p)[int((t0 + float(d)) * 48000): int((t1 + float(d)) * 48000)]
        y = y / y.pow(2).mean().sqrt()
        s = torch.stft(y, nfft, 24, window=torch.hann_window(nfft), return_complex=True).abs().clamp_min(1e-5).log10() * 20
        s = s[: int(fmax / 48000 * nfft)].flip(0); s = ((s - (s.max() - 70)) / 70).clamp(0, 1)
        cols += [torch.nn.functional.interpolate(s[None, None], size=(430, 520), mode="bilinear")[0, 0], torch.ones(430, 6)]
    rows += [torch.cat(cols, 1), torch.ones(6, 526 * len(sys.argv[4:]))]
a = (torch.cat(rows, 0).numpy() * 255).astype(np.uint8); h, w = a.shape
ch = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
open(sys.argv[3], "wb").write(b"\x89PNG\r\n\x1a\n" + ch(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)) + ch(b"IDAT", zlib.compress(b"".join(b"\0" + a[i].tobytes() for i in range(h)))) + ch(b"IEND", b""))
