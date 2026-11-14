"""Audio I/O helpers. Kept dependency-light (numpy + soundfile) because this
module is also imported inside third-party system environments."""

from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf


def resample(wav: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return wav.astype(np.float32, copy=False)
    try:
        import soxr

        return soxr.resample(wav, sr_in, sr_out, quality="HQ").astype(np.float32)
    except ImportError:
        from scipy.signal import resample_poly

        g = gcd(sr_in, sr_out)
        return resample_poly(wav, sr_out // g, sr_in // g).astype(np.float32)


def load(path: str | Path, sr: int | None = None) -> tuple[np.ndarray, int]:
    """Load as mono float32, optionally resampled to `sr`."""
    wav, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr is not None and sr != file_sr:
        wav = resample(wav, file_sr, sr)
        file_sr = sr
    return wav, file_sr


def save(path: str | Path, wav: np.ndarray, sr: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(wav, -1.0, 1.0), sr, subtype="PCM_16")


def duration(path: str | Path) -> float:
    info = sf.info(str(path))
    return info.frames / info.samplerate
