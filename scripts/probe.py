"""Measurements used while developing causalvc, on any 48 kHz-loadable recording.

    PYTHONPATH=$PWD .venv/bin/python scripts/probe.py TARGET.wav OUTPUT.wav [OUTPUT2.wav ...]

Compares converted recordings with a long recording of the real target voice:
speaker similarity, predicted naturalness, pitch range, harmonic clarity, tone
per band, and (for a mix of microphone and model output) the echo delay.
Set DEV to choose the GPU (default cuda:1).
"""
import os, sys
import numpy as np, soundfile as sf, torch, torchaudio

DEV = os.environ.get("DEV", "cuda:1")
BANDS = ((300, 1000), (1000, 2000), (2000, 4000), (4000, 8000))
EDGES = [80, 250, 500, 1000, 2000, 4000, 8000, 16000]
_rm = _cp = None


def load48(p):
    w, sr = sf.read(p, dtype="float32", always_2d=True)
    x = torch.from_numpy(w[:, 0])
    return torchaudio.functional.resample(x, sr, 48000) if sr != 48000 else x


def rmvpe_f0(x):
    """[N] at 48 kHz -> F0 per 10 ms frame (0 unvoiced), on DEV."""
    global _rm
    if _rm is None:
        from causalvc.pitch import Rmvpe
        _rm = Rmvpe(DEV)
    with torch.no_grad():
        return _rm(torchaudio.functional.resample(x.to(DEV), 48000, 16000)[None])[0]


def clarity(x):
    """Harmonic clarity in dB per band: energy on harmonics against energy between them,
    on pitch-stable voiced frames. Returns (list, frames used) or (None, 0)."""
    global _cp
    if _cp is None:
        from causalvc.pitch_causal import load_pitch
        _cp = load_pitch("runs/models/pitch-la2/pitch.pt", DEV)
    x = x.to(DEV)
    with torch.no_grad():
        f0 = _cp(torchaudio.functional.resample(x, 48000, 16000)[None])[0]
    S = torch.stft(x, 4096, 480, window=torch.hann_window(4096, device=DEV), return_complex=True).abs().pow(2)
    T = min(len(f0), S.shape[1]); fr = torch.arange(S.shape[0], device=DEV) * 48000 / 4096
    f = f0[:T]; win = f.unfold(0, 5, 1)
    ok = (win.min(1).values > 0) & (win.max(1).values / win.min(1).values.clamp_min(1) < 1.015)
    ts = torch.nonzero(ok)[:, 0] + 2
    if len(ts) < 10:
        return None, 0
    fc = f[ts]

    def ratio(fcs, lo, hi):
        ph = (fr[None, None] / fcs[..., None]) % 1.0; band = ((fr >= lo) & (fr < hi))[None, None]
        on = (band & ((ph < 0.15) | (ph > 0.85))).float(); off = (band & (ph > 0.35) & (ph < 0.65)).float()
        Sp = S[:, ts].T[:, None]
        return 10 * torch.log10(((Sp * on).sum(-1) / on.sum(-1).clamp_min(1)) / ((Sp * off).sum(-1) / off.sum(-1).clamp_min(1)).clamp_min(1e-12) + 1e-12)

    c1 = fc[:, None] * (1 + torch.arange(-20, 21, 2, device=DEV) / 1000)[None]
    fc = c1[torch.arange(len(ts)), ratio(c1, 1000, 4000).argmax(1)]
    c2 = fc[:, None] * (1 + torch.arange(-6, 7, device=DEV) / 2000)[None]
    fc = c2[torch.arange(len(ts)), ratio(c2, 4000, 8000).argmax(1)]
    return [float(ratio(fc[:, None], lo, hi).mean()) for lo, hi in BANDS], len(ts)


def tone(ws):
    """Long-term voiced spectrum per band (EDGES), in dB relative to the 80 Hz-16 kHz total."""
    fr = torch.arange(1025, device=DEV) * 48000 / 2048; tot, n = 0, 0
    for w in ws:
        yd = w.to(DEV); f0 = rmvpe_f0(yd)
        Sp = torch.stft(yd, 2048, 480, window=torch.hann_window(2048, device=DEV), return_complex=True).abs().pow(2)
        T = min(Sp.shape[1], len(f0)); v = f0[:T] > 0
        tot = tot + Sp[:, :T][:, v].sum(1); n += int(v.sum())
    S = tot / n
    b = torch.stack([S[(fr >= a) & (fr < c)].sum() for a, c in zip(EDGES, EDGES[1:])])
    return (10 * torch.log10(b) - 10 * torch.log10(S[(fr >= 80) & (fr < 16000)].sum())).cpu().numpy()


def echo_delay(x):
    """For a mix of a signal and a delayed copy (mic plus model): delay in ms and peak strength."""
    S = torch.stft(x, 1024, 48, window=torch.hann_window(1024), return_complex=True).abs()
    L = torch.log(S + 1e-4); fl = (L[:, 1:] - L[:, :-1]).clamp_min(0).sum(0).numpy(); fl -= fl.mean()
    a = np.correlate(fl, fl, "full")[len(fl) - 1:]; a /= a[0]
    i = 25 + int(np.argmax(a[25:350]))
    return i, float(a[i])


if __name__ == "__main__":
    from vcbench.metrics.models import MosPredictor, SpeakerEmbedder
    emb, mos = SpeakerEmbedder(DEV), MosPredictor(DEV)
    r16 = lambda x: torchaudio.functional.resample(x.cpu(), 48000, 16000).numpy()
    tgt = load48(sys.argv[1]); n = 48000 * 8
    real = [tgt[i:i + n] for i in np.linspace(0, len(tgt) - n, 24).astype(int)]
    cen = emb.centroid([r16(x) for x in real[0::2]]); ceil = np.mean([emb.embed(r16(x)) @ cen for x in real[1::2]])
    rb = tone(real); hr = np.mean([h for h in (clarity(x)[0] for x in real) if h], 0)
    fr = torch.cat([f[f > 0] for f in map(rmvpe_f0, real[:8])])
    print(f"real target: median f0 {float(fr.median()):.0f} Hz (10-90%: {float(fr.quantile(.1)):.0f}-{float(fr.quantile(.9)):.0f}), real-vs-real sim {ceil:.3f}, "
          f"utmos {np.mean([mos.score(r16(x)) for x in real[:8]]):.2f}, clarity {np.round(hr, 1)}")
    for p in sys.argv[2:]:
        y = load48(p); f = rmvpe_f0(y); f = f[f > 0]; lag, v = echo_delay(y); c = clarity(y)[0]
        print(f"{os.path.basename(p)}: sim {emb.embed(r16(y)) @ cen:.3f}, utmos {mos.score(r16(y)):.2f}, median f0 {float(f.median()):.0f} Hz "
              f"(10-90%: {float(f.quantile(.1)):.0f}-{float(f.quantile(.9)):.0f}), clarity {np.round(c, 1) if c else None}, "
              f"tone minus real {np.round(tone([y]) - rb, 1)}, echo {lag} ms ({v:.2f})")
