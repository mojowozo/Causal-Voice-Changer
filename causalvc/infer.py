"""Whole-file conversion with the causal encoder and decoder.

Both networks are causal, so this is what a streaming runtime would output; the
pitch track is the exception while it still comes from RMVPE.
"""

from __future__ import annotations

import torch
import torchaudio

from causalvc.decoder.model import FRAME, SR, Decoder, loudness, tone
from causalvc.encoder.model import StreamingEncoder
from causalvc.pitch import Rmvpe


def load_encoder(path: str, device: str) -> StreamingEncoder:
    ckpt = torch.load(path, map_location="cpu")
    enc = StreamingEncoder(**ckpt["cfg"])
    enc.load_state_dict(ckpt["model"])
    return enc.to(device).eval().requires_grad_(False)


@torch.no_grad()
def extract(enc: StreamingEncoder, rmvpe: Rmvpe, wav48: torch.Tensor):
    """[B, N] at 48 kHz -> content [B, T/2, C], f0 [B, T], loudness [B, T]."""
    wav16 = torchaudio.functional.resample(wav48, SR, 16000)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=wav48.is_cuda and isinstance(enc, StreamingEncoder)):
        content = enc.encode(wav16).float()
    return content, rmvpe(wav16), loudness(wav48)


class CausalVC:
    def __init__(self, decoder: str, device: str = "cuda:0", encoder: str | None = None, pitch: str | None = None,
                 index: float = 0.0, index_k: int = 8, manifest: str = "data/eval/manifest.json", hsmooth: int = 0):
        self.hsmooth = hsmooth  # diagnostic: moving average (2k+1 frames) over the decoder's 100 Hz features
        self.index, self.index_k, self.manifest, self.banks = index, index_k, manifest, {}
        ckpt = torch.load(decoder, map_location="cpu")
        self.device, self.speakers = device, ckpt["speakers"]
        if encoder == "teacher":  # diagnostic: the non-causal teacher's features instead of the student's
            from causalvc.encoder.train import load_teacher

            class _Teacher:
                def __init__(self):
                    self.m = load_teacher(device)

                def encode(self, wav16):
                    return self.m(wav16.float()).last_hidden_state

            self.enc = _Teacher()
        else:
            from causalvc.voice import encoder_path

            self.enc = load_encoder(encoder_path(encoder or ckpt.get("encoder")), device)
        self.dec = Decoder(len(self.speakers), **ckpt["cfg"])
        self.dec.load_state_dict(ckpt["generator"])
        self.dec.to(device).eval()
        if pitch:  # causal tracker instead of whole-file RMVPE
            from causalvc.pitch_causal import load_pitch

            self.rmvpe = load_pitch(pitch, device)
        else:
            self.rmvpe = Rmvpe(device)

    @torch.no_grad()
    def convert(self, wav48: torch.Tensor, speaker: str, semitones: float = 0.0) -> torch.Tensor:
        """Mono [N] at 48 kHz in, same length out."""
        n = wav48.shape[-1]
        x = torch.nn.functional.pad(wav48.to(self.device)[None], (0, (-n) % (2 * FRAME)))
        content, f0, loud = extract(self.enc, self.rmvpe, x)
        if self.index > 0:  # pull each content frame toward the target's own nearest frames
            near = self.lookup(content[0], self.bank(speaker))
            content = (1 - self.index) * content + self.index * near[None]
        spk = torch.tensor([self.speakers.index(speaker)], device=self.device)
        f0 = f0 * 2.0 ** (semitones / 12.0)
        h, g = self.dec.features(content, f0, loud, spk, tone=tone(x) if self.dec.tone is not None else None)
        if self.hsmooth:
            k = self.hsmooth
            h = torch.nn.functional.avg_pool1d(torch.nn.functional.pad(h, (k, k), mode="replicate"), 2 * k + 1, 1)
        y = self.dec.generate(h, g, self.dec.source(f0[:, : h.shape[-1]]))[0, 0]
        return torch.nn.functional.pad(y, (0, max(0, n - len(y))))[:n]

    @torch.no_grad()
    def bank(self, speaker: str, max_frames: int = 120000) -> torch.Tensor:
        """Content frames of the target's own training audio, [M, C]."""
        if speaker not in self.banks:
            import json

            import soundfile as sf

            feats, n = [], 0
            for path in json.load(open(self.manifest))["targets"][speaker]["train"]:
                wav, sr = sf.read(path, dtype="float32", always_2d=True)
                w16 = torchaudio.functional.resample(torch.from_numpy(wav[:, 0]), sr, 16000).to(self.device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=isinstance(self.enc, StreamingEncoder)):
                    feats.append(self.enc.encode(w16[None]).float()[0])
                n += len(feats[-1])
                if n >= max_frames:
                    break
            self.banks[speaker] = torch.cat(feats)[:max_frames]
        return self.banks[speaker]

    def lookup(self, x: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
        """[T, C] -> the similarity-weighted mean of each frame's k nearest bank frames."""
        sim = torch.nn.functional.normalize(x, dim=-1) @ torch.nn.functional.normalize(bank, dim=-1).T
        top, idx = sim.topk(self.index_k, dim=-1)
        w = torch.softmax(top / 0.05, dim=-1)
        return (bank[idx] * w[..., None]).sum(1)
