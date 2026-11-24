"""Transparent output peak limiter and soft clipper.

``soft_clip`` rounds samples above ``knee`` smoothly toward full scale with zero
latency.  ``PeakLimiter`` handles loud speech peaks via a smooth gain envelope
(bit-exact below -1 dBFS, no memoryless odd-harmonic distortion on loud vowels)
backed by a -0.5 dBFS soft-clip safety ceiling so samples never exceed +/-1.0.
"""

import math

import numpy as np
import torch

KNEE_DB = -6.0
KNEE = 10 ** (KNEE_DB / 20)  # about 0.501
_HEADROOM = 1.0 - KNEE
SAFETY_KNEE = 10 ** (-1.0 / 20)  # -1.0 dBFS (~0.891) for input safety & limiter threshold
CEILING_KNEE = 10 ** (-0.5 / 20)  # -0.5 dBFS (~0.944) final safety backstop


def soft_clip(audio, knee: float = KNEE):
    """Soft-clip a torch tensor or numpy array above ``knee``; stays within +/-1."""
    headroom = 1.0 - float(knee)
    if isinstance(audio, torch.Tensor):
        magnitude = audio.abs()
        over = torch.clamp(magnitude - knee, min=0.0)
        shaped = knee + headroom * torch.tanh(over / headroom)
        return torch.where(magnitude > knee, torch.sign(audio) * shaped, audio)
    audio = np.asarray(audio)
    magnitude = np.abs(audio)
    over = np.maximum(magnitude - knee, 0.0)
    shaped = knee + headroom * np.tanh(over / headroom)
    return np.where(magnitude > knee, np.sign(audio) * shaped, audio).astype(audio.dtype, copy=False)


class PeakLimiter:
    """Zero-latency smooth gain-envelope peak limiter with a -0.5 dBFS soft-clip safety backstop.

    Audio whose peaks stay below ``threshold`` (-1.0 dBFS by default) passes through bit-exact
    with zero added harmonics or aliasing.  Peaks above ``threshold`` smoothly pull down a
    gain envelope (fast attack, brief hold, 60 ms release) instead of flat-topping or applying
    a low-knee waveshaper across every loud vowel.
    """

    def __init__(
        self,
        samplerate: int = 48000,
        threshold: float = SAFETY_KNEE,
        release_seconds: float = 0.060,
        hold_seconds: float = 0.020,
        step_seconds: float = 0.0005,
    ):
        self.samplerate = int(samplerate)
        self.threshold = float(threshold)
        self.step = max(1, round(step_seconds * self.samplerate))
        self.hold_samples = max(1, round(hold_seconds * self.samplerate))
        self.release_coeff = math.exp(-self.step / max(1.0, release_seconds * self.samplerate))
        self.reset()

    def reset(self) -> None:
        self.gain = 1.0
        self.hold_remaining = 0

    def process(self, block: np.ndarray) -> np.ndarray:
        audio = np.asarray(block, dtype=np.float32)
        if audio.size == 0:
            return audio
        # Fast path: completely transparent when idle and below threshold.
        peak_all = float(np.max(np.abs(audio)))
        if self.gain >= 0.999999 and peak_all <= self.threshold:
            self.gain = 1.0
            return audio

        out = np.empty_like(audio)
        pos = 0
        total = audio.shape[0]
        step = self.step
        thr = self.threshold
        rel = self.release_coeff
        while pos < total:
            end = min(pos + step, total)
            chunk = audio[pos:end]
            n = end - pos
            pk = float(np.max(np.abs(chunk)))
            target = min(1.0, thr / pk) if pk > thr else 1.0
            prev_g = self.gain
            if target < prev_g:
                new_g = target
                self.hold_remaining = self.hold_samples
            elif pk * prev_g >= thr * 0.98:
                new_g = prev_g
                self.hold_remaining = self.hold_samples
            elif self.hold_remaining > 0:
                new_g = prev_g
                self.hold_remaining = max(0, self.hold_remaining - n)
            else:
                coeff = rel if n == step else (rel ** (n / step))
                new_g = 1.0 - (1.0 - prev_g) * coeff
                if new_g > 0.99999:
                    new_g = 1.0
            self.gain = new_g
            if prev_g == 1.0 and new_g == 1.0:
                out[pos:end] = chunk
            elif prev_g == new_g:
                out[pos:end] = chunk * np.float32(new_g)
            else:
                ramp = np.linspace(prev_g, new_g, n, endpoint=True, dtype=np.float32)
                out[pos:end] = chunk * ramp
            pos = end
        # Final safety ceiling for any intra-subframe transient spike above -0.5 dBFS.
        if float(np.max(np.abs(out))) > CEILING_KNEE:
            return soft_clip(out, knee=CEILING_KNEE)
        return out
