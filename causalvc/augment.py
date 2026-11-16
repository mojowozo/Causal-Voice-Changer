"""Microphone-style degradation for training inputs, on the GPU.

The clean signal stays the training target; only the model's input goes
through this. Each example independently gets a random tone curve, and with
some probability room echo, coloured background noise, a low-pass and clipping.
Everything is time-aligned with the clean signal (the echo starts at lag 0).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def degrade(wav: torch.Tensor, sr: int = 16000, strength: float = 1.0) -> torch.Tensor:
    """[B, N] -> [B, N]."""
    b, n = wav.shape
    dev = wav.device
    rand = lambda lo, hi: torch.empty(b, 1, device=dev).uniform_(lo, hi)  # noqa: E731
    coin = lambda p: (torch.rand(b, 1, device=dev) < p).float()  # noqa: E731
    tail = int(0.6 * sr)
    m = n + tail
    bins = m // 2 + 1
    freq = torch.linspace(0, sr / 2, bins, device=dev)[None]

    # Tone: a smooth random curve of up to +-8 dB over 8 control points.
    curve = torch.empty(b, 1, 8, device=dev).uniform_(-8, 8) * strength
    eq = 10 ** (F.interpolate(curve, size=bins, mode="linear", align_corners=True)[:, 0] / 20)
    # Low-pass (cheap mics, codecs): 12 dB/octave above a random corner.
    corner = rand(3000, 7500)
    lp = coin(0.25) * (1 / (1 + (freq / corner) ** 2) - 1) + 1
    # Room: decaying noise after a unit direct path.
    t = torch.arange(tail, device=dev)[None] / sr
    rir = torch.randn(b, tail, device=dev) * torch.exp(-6.9 * t / rand(0.15, 0.7))
    rir = rir / rir.pow(2).sum(1, keepdim=True).sqrt() * 10 ** (rand(-22, -4) / 20) * coin(0.5 * strength)
    rir[:, 0] = 1.0
    spec = torch.fft.rfft(wav, m) * eq * lp * torch.fft.rfft(rir, m)
    out = torch.fft.irfft(spec, m)[:, :n]

    # Background noise with a random spectral tilt, 8-40 dB below the speech.
    tilt = (freq[:, : n // 2 + 1].clamp_min(20) / 1000) ** (-rand(0.0, 1.5))
    noise = torch.fft.irfft(torch.fft.rfft(torch.randn(b, n, device=dev)) * tilt, n)
    rms = lambda x: x.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-5)  # noqa: E731
    out = out + noise / rms(noise) * rms(out) * 10 ** (-rand(8, 40) / 20) * coin(0.7 * strength)

    # Clipping.
    peak = out.abs().amax(1, keepdim=True).clamp_min(1e-5)
    limit = torch.where(coin(0.1 * strength) > 0, rand(0.25, 0.8), torch.full_like(peak, 10.0)) * peak
    return torch.maximum(torch.minimum(out, limit), -limit)


@torch.no_grad()
def add_noise(wav: torch.Tensor, share: float = 0.6, silent: float = 0.12, sr: int = 48000) -> tuple[torch.Tensor, torch.Tensor]:
    """Non-voice sounds a microphone picks up, for teaching a decoder to stay quiet through them.

    [B, N] clean -> (input, target). For `share` of the examples the input gains steady background
    noise and/or short bursts (clicks, snaps, claps, thumps) while the target stays clean; for `silent`
    of them the voice is removed as well, so the input is noise alone and the target is silence."""
    b, n = wav.shape
    dev = wav.device
    inp, target = wav.clone(), wav.clone()
    freq = torch.fft.rfftfreq(n, 1 / sr, device=dev).clamp_min(20)
    u = lambda lo, hi: float(torch.empty(1).uniform_(lo, hi))  # noqa: E731
    for i in range(b):
        r = u(0, 1)
        if r >= share:
            continue
        if r < silent:
            inp[i], target[i] = 0.0, 0.0
        if r < silent or u(0, 1) < 0.6:  # steady noise with a random tilt, far below ordinary speech
            x = torch.fft.irfft(torch.fft.rfft(torch.randn(n, device=dev)) * freq ** (-u(0.0, 1.5)), n)
            inp[i] += x / x.pow(2).mean().sqrt() * 10 ** (u(-65, -35) / 20)
        if u(0, 1) < 0.75:
            for _ in range(int(u(1, 6))):  # bursts: 3-90 ms of band-limited noise with a sharp start and a decaying tail
                k = int(u(0.003, 0.09) * sr)
                at = int(u(0, n - k))
                burst = torch.randn(k, device=dev) * torch.exp(-torch.arange(k, device=dev) / (k / u(3, 8)))
                f = torch.fft.rfftfreq(k, 1 / sr, device=dev)
                lo = 10 ** u(1.6, 3.5)
                spec = torch.fft.rfft(burst)
                spec[(f < lo) | (f > min(lo * u(2, 10), 14000.0))] = 0
                burst = torch.fft.irfft(spec, k)
                inp[i, at : at + k] += burst / burst.abs().max().clamp_min(1e-6) * 10 ** (u(-35, -3) / 20)
    return inp.clamp(-1.0, 1.0), target
