"""RVC whole-file inference with ContentVec swapped for the causal streaming encoder.

Isolates one question: how much quality, similarity and leakage change when the
content features come from a causal model instead of ContentVec, with the
same RVC voices, pitch tracker and decoder. Offline only (`--blocks offline`):
the encoder is causal, so its whole-file output equals its streamed output.

    PYTHONPATH=$PWD /mnt/ssd/Applio-3.7.1/.venv/bin/python -m vcbench.convert \
        --system vcbench.systems.rvc_causal_enc:RvcCausalEnc \
        --system-args '{"encoder":"runs/models/enc-la2/encoder.pt"}' --run rvc-enc-la2 --blocks offline
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from vcbench.systems.rvc_applio import RvcApplio


class _AsHubert(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, wav: torch.Tensor) -> dict:
        feats = self.encoder.encode(wav.float())
        n = (wav.shape[-1] - 400) // 320 + 1  # HuBERT's frame count
        return {"last_hidden_state": feats[:, :n]}


class RvcCausalEnc(RvcApplio):
    name = "rvc-causal-enc"

    def __init__(self, encoder: str, **kwargs):
        super().__init__(**kwargs)
        self.encoder_path = str(Path(encoder).resolve())

    def set_block_size(self, n_samples: int) -> None:
        raise NotImplementedError("offline only: use --blocks offline")

    def convert_offline(self, wav: np.ndarray) -> np.ndarray:
        if self._offline is None:
            self._enter_applio()
            from rvc.infer.infer import VoiceConverter

            from causalvc.encoder.model import StreamingEncoder

            ckpt = torch.load(self.encoder_path, map_location="cpu")
            enc = StreamingEncoder(**ckpt["cfg"])
            enc.load_state_dict(ckpt["model"])
            self._offline = VoiceConverter()
            dev = self._offline.config.device
            self._offline.hubert_model = _AsHubert(enc).to(dev).float().eval()
            self._offline.last_embedder_model = self.embedder
        return super().convert_offline(wav)
