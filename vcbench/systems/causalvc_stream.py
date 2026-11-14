"""The new model run block by block, as a realtime application would.

    PYTHONPATH=$PWD .venv/bin/python -m vcbench.convert \
        --system vcbench.systems.causalvc_stream:CausalVCStream \
        --system-args '{"decoder":"runs/models/dec-s0/decoder.pt","pitch":"runs/models/pitch-la2/pitch.pt"}' \
        --run next-stream --blocks 20,40
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from vcbench.streaming import VCSystem, Voice


class CausalVCStream(VCSystem):
    name = "causalvc-stream"
    io_sr = 48000

    def __init__(self, decoder: str, pitch: str, device: str = "cuda:0", encoder: str | None = None, level: float | None = None, fast: bool = False):
        self.fast = fast  # use the cached runtime (causalvc.fast), which computes only what is new each block
        self.level = level  # optional automatic input level target in dB; None leaves the input as is
        self.args = (str(Path(decoder).resolve()), str(Path(pitch).resolve()), device, encoder)
        self.device = device
        self.vc = None
        self.voice, self.semitones = None, 0

    def load_voice(self, voice: Voice) -> None:
        if self.vc is None:
            from causalvc.fast import streaming
            from causalvc.stream import StreamingVC

            self.vc = streaming(*self.args, level=self.level) if self.fast else StreamingVC(*self.args, level=self.level)
        if voice.id not in self.vc.speakers:
            raise RuntimeError(f"{voice.id} is not a speaker of {self.args[0]}")
        self.voice = voice.id
        self.vc.set_voice(self.voice, self.semitones)

    def set_pitch_shift(self, semitones: int) -> None:
        self.semitones = int(semitones)
        if self.vc is not None and self.voice is not None:
            self.vc.set_voice(self.voice, self.semitones)

    def set_block_size(self, n_samples: int) -> None:
        if n_samples % 960:
            raise ValueError("block size must be a multiple of 20 ms")

    def reset(self) -> None:
        self.vc.reset()

    def process_block(self, block: np.ndarray) -> np.ndarray:
        return self.vc.process(torch.from_numpy(block)).cpu().numpy().astype(np.float32)

    def reported_latency_ms(self) -> float:
        from causalvc.stream import DELAY

        return (self.vc.delay if self.vc is not None else DELAY) * 10.0

    def sync(self) -> None:
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
