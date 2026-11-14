"""Sanity-check systems that validate the harness itself.

Passthrough: output == input, optionally delayed. Expected scores: speaker
similarity to *source* high and to target low, WER at the ASR floor, F0
correlation ~1, measured delay == delay_ms, block ripple ~0.

With jitter_ms > 0 each block is read with its own random extra delay and
crossfaded into the previous one over 2 ms. That is the failure mode of chunked
converters (independently generated chunks that do not line up), so it gives
the boundary-artifact metric something realistic to find.
"""

from __future__ import annotations

import numpy as np

from vcbench.streaming import VCSystem, Voice


class Passthrough(VCSystem):
    name = "passthrough"

    def __init__(self, io_sr: int = 24000, delay_ms: float = 0.0, jitter_ms: float = 0.0):
        self.io_sr = io_sr
        self.delay = int(round(io_sr * delay_ms / 1000.0))
        self.delay_ms = delay_ms
        self.jitter = int(round(io_sr * jitter_ms / 1000.0))
        self.xfade = int(io_sr * 0.002)
        self.reset()

    def load_voice(self, voice: Voice) -> None:
        pass

    def reset(self) -> None:
        self.hist = np.zeros(self.delay + self.jitter, dtype=np.float32)
        self.rng = np.random.default_rng(0)
        self.prev_off = 0

    def process_block(self, block: np.ndarray) -> np.ndarray:
        n, keep = len(block), self.delay + self.jitter
        self.hist = np.concatenate([self.hist, block])[-(keep + n) :]
        read = lambda off: self.hist[len(self.hist) - n - self.delay - off : len(self.hist) - self.delay - off]
        off = int(self.rng.integers(0, self.jitter + 1)) if self.jitter else 0
        out = read(off).copy()
        if off != self.prev_off:
            x = min(self.xfade, n)
            ramp = np.linspace(0.0, 1.0, x, dtype=np.float32)
            out[:x] = read(self.prev_off)[:x] * (1 - ramp) + out[:x] * ramp
        self.prev_off = off
        return out

    def convert_offline(self, wav: np.ndarray) -> np.ndarray:
        return wav.copy()

    def reported_latency_ms(self) -> float:
        return self.delay_ms
