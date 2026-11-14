"""Signal-level metrics: delay, block-boundary artifacts, streaming-vs-offline drift."""

from __future__ import annotations

import numpy as np
from scipy.signal import correlate, stft


def _envelope(wav: np.ndarray, sr: int, hop_ms: float = 1.0, win_ms: float = 5.0) -> np.ndarray:
    hop = int(sr * hop_ms / 1000)
    win = int(sr * win_ms / 1000)
    power = np.convolve(wav.astype(np.float64) ** 2, np.ones(win) / win, mode="same")
    return np.log(power[::hop] + 1e-8)


def estimate_delay_ms(src: np.ndarray, out: np.ndarray, sr: int, max_ms: float = 800.0) -> float:
    """Delay of `out` relative to `src` from energy-envelope cross-correlation (1 ms resolution).

    Envelopes survive voice conversion (same syllable timing) where waveform
    correlation would not.
    """
    a, b = _envelope(src, sr), _envelope(out, sr)
    n = max(len(a), len(b))
    a = np.pad(a - a.mean(), (0, n - len(a)))
    b = np.pad(b - b.mean(), (0, n - len(b)))
    xc = correlate(b, a, mode="full")
    lags = np.arange(-n + 1, n)
    keep = (lags >= -20) & (lags <= max_ms)
    return float(lags[keep][np.argmax(xc[keep])])


def align(out: np.ndarray, delay_ms: float, sr: int, length: int) -> np.ndarray:
    """Remove `delay_ms` from the front of `out` and fit to `length` samples."""
    d = int(round(sr * delay_ms / 1000))
    out = out[d:] if d >= 0 else np.pad(out, (-d, 0))
    return np.pad(out, (0, max(0, length - len(out))))[:length]


def block_ripple(out: np.ndarray, sr: int, block_size: int, n_bands: int = 8) -> tuple[float, float]:
    """Band-energy modulation locked to the block rate: (variance, noise floor) in dB².

    Chunked converters leave a signature at every block edge (partial
    cancellation in the crossfade, level or timbre steps between chunks). A
    single edge is buried under normal speech variation, so average the
    short-time band energies as a function of position within the block:
    speech is unrelated to block phase and averages out, the signature does
    not. The same fold at unrelated periods estimates the noise floor. One clip
    is too short for a stable estimate, so the caller pools both numbers over
    the test set and reports sqrt(mean variance - mean floor).
    """
    from scipy.ndimage import uniform_filter1d

    hop, win = max(1, sr // 1000), int(sr * 0.004)
    freqs, _, z = stft(out, sr, nperseg=win, noverlap=win - hop, boundary=None, padded=False)
    power = np.abs(z) ** 2
    edges = np.geomspace(100.0, sr / 2, n_bands + 1)
    bands = np.stack([power[(freqs >= lo) & (freqs < hi)].sum(axis=0) for lo, hi in zip(edges[:-1], edges[1:])])
    level = 10 * np.log10(bands + 1e-10)
    total = 10 * np.log10(power.sum(axis=0) + 1e-10)
    audible = total > total.max() - 35
    frames_per_block = block_size / hop
    if audible.sum() < 20 * frames_per_block or frames_per_block < 3:
        return float("nan"), float("nan")
    # Remove the speech envelope; a moving average over exactly one block
    # leaves any block-periodic component untouched.
    resid = level - uniform_filter1d(level, size=max(3, int(round(frames_per_block))), axis=1, mode="nearest")
    centers = np.arange(level.shape[1]) * hop + win / 2

    def ripple(period: float) -> float:
        n_bins = int(min(round(period / hop), 32))
        phase = np.floor((centers % period) / period * n_bins).astype(int)
        prof = np.stack([resid[:, audible & (phase == k)].mean(axis=1) for k in range(n_bins)], axis=1)
        return float(np.mean((prof - prof.mean(axis=1, keepdims=True)) ** 2))

    floor = np.mean([ripple(block_size * r) for r in (0.8713, 1.1374, 1.2931)])
    return ripple(float(block_size)), float(floor)


def logmel(wav: np.ndarray, sr: int, n_mels: int = 80) -> np.ndarray:
    win, hop = int(sr * 0.025), int(sr * 0.010)
    _, _, z = stft(wav, sr, nperseg=win, noverlap=win - hop, nfft=1 << (win - 1).bit_length(), boundary=None, padded=False)
    power = np.abs(z) ** 2
    freqs = np.linspace(0, sr / 2, power.shape[0])
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    edges = 700 * (10 ** (np.linspace(mel(50), mel(sr / 2), n_mels + 2) / 2595) - 1)
    fb = np.zeros((n_mels, len(freqs)))
    for i in range(n_mels):
        lo, mid, hi = edges[i : i + 3]
        fb[i] = np.clip(np.minimum((freqs - lo) / (mid - lo), (hi - freqs) / (hi - mid)), 0, None)
    return 10 * np.log10(fb @ power + 1e-8)


def mel_distance_db(a: np.ndarray, b: np.ndarray, sr: int) -> float:
    """Mean absolute log-mel difference in dB over frames where either is audible."""
    n = min(len(a), len(b))
    ma, mb = logmel(a[:n], sr), logmel(b[:n], sr)
    level = np.maximum(ma.max(axis=0), mb.max(axis=0))
    audible = level > level.max() - 50
    return float(np.abs(ma - mb)[:, audible].mean())
