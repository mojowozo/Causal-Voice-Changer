"""The new model (causal encoder + causal decoder), scored through the harness.

    PYTHONPATH=$PWD .venv/bin/python -m vcbench.convert \
        --system vcbench.systems.causalvc_offline:CausalVCOffline \
        --system-args '{"decoder":"runs/models/dec-s0/decoder.pt"}' --run next-s0 --blocks offline

Offline only for now. Encoder and decoder are causal, so their whole-file
output is what a streaming runtime would produce; the pitch track still comes
from whole-file RMVPE, which a streaming runtime would not have. Target voices
must be speakers of the decoder checkpoint (the base is trained with the
harness's target voices included), so there is nothing to build.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from vcbench.streaming import VCSystem, Voice


class CausalVCOffline(VCSystem):
    name = "causalvc"
    io_sr = 48000

    def __init__(self, decoder: str, device: str = "cuda:0", encoder: str | None = None, pitch: str | None = None, index: float = 0.0, hsmooth: int = 0):
        self.index, self.hsmooth = index, hsmooth
        self.decoder, self.encoder, self.device, self.pitch = str(Path(decoder).resolve()), encoder, device, pitch
        self.vc = None
        self.voice = None
        self.semitones = 0

    def load_voice(self, voice: Voice) -> None:
        if self.vc is None:
            from causalvc.infer import CausalVC

            self.vc = CausalVC(self.decoder, self.device, self.encoder, self.pitch, self.index, hsmooth=self.hsmooth)
        if voice.id not in self.vc.speakers:
            raise RuntimeError(f"{voice.id} is not a speaker of {self.decoder}")
        self.voice = voice.id

    def set_pitch_shift(self, semitones: int) -> None:
        self.semitones = int(semitones)

    def set_block_size(self, n_samples: int) -> None:
        raise NotImplementedError("offline only: use --blocks offline")

    def process_block(self, block: np.ndarray) -> np.ndarray:
        raise NotImplementedError("offline only: use --blocks offline")

    def convert_offline(self, wav: np.ndarray) -> np.ndarray:
        y = self.vc.convert(torch.from_numpy(wav), self.voice, self.semitones)
        return y.cpu().numpy().astype(np.float32)

    def sync(self) -> None:
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
