"""Block-by-block conversion with the causal encoder, pitch tracker and decoder.

Each block, the encoder and pitch tracker run over the last few seconds of
input and only their newly final frames are kept; the decoder then generates
just the new audio from a short history of those frames. Output trails input by
`DELAY` frames (50 ms): the encoder's 40 ms lookahead and the pitch tracker's
52 ms, rounded up to the 10 ms frame clock. Blocks must be multiples of 20 ms.
"""

from __future__ import annotations

import math

import torch
import torchaudio

from causalvc.decoder.model import FRAME, SR, Decoder, loudness, tone
from causalvc.infer import load_encoder
from causalvc.voice import encoder_path
from causalvc.pitch_causal import load_pitch

DELAY = 5  # 100 Hz frames between the newest input and the newest output
HIST = 128  # frames of features kept for the decoder
GEN = 32  # frames of context the waveform generator is given
F0_RANGE = (50.0, 1100.0)  # Hz: the span of the decoder's pitch buckets (Decoder.pitch_bucket)


def map_f0(f0: torch.Tensor, ratio: float, offset: float = 0.0, ceiling: float = 0.0, keep: float = 1.0) -> torch.Tensor:
    """The pitch the voice is given for the tracked pitch ``f0`` (Hz, 0 unvoiced).

    ``f0 * ratio + offset``, then, with a ``ceiling`` (Hz, 0 = none), only ``keep`` (0-1) of the rise
    above it: in semitones, so notes below the ceiling keep their intervals (singing stays in tune)
    and only peaks above it are pulled down (keep 0 stops them at the ceiling). Both apply to voiced
    frames only (unvoiced ones stay 0) and keep them within the decoder's pitch range; without
    either this is the plain multiplication."""
    if offset == 0.0 and ceiling <= 0.0:
        return f0 * ratio
    y = (f0 * ratio + offset).clamp(*F0_RANGE)
    if ceiling > 0.0:
        y = torch.where(y > ceiling, ceiling * (y / ceiling) ** max(0.0, min(1.0, keep)), y)
    return torch.where(f0 > 0, y, torch.zeros_like(f0))


class StreamingVC:
    def __init__(self, decoder: str, pitch: str, device: str = "cuda:0", encoder: str | None = None, window: float = 4.0,
                 level: float | None = None):
        self.level_target = level  # None (default) leaves the input as is; a dB RMS value turns on a slow automatic gain toward it
        ckpt = torch.load(decoder, map_location="cpu")
        self.device, self.speakers = device, ckpt["speakers"]
        self.enc = load_encoder(encoder_path(encoder or ckpt.get("encoder")), device)
        self.pitch = load_pitch(pitch, device)
        self.dec = Decoder(len(self.speakers), **ckpt["cfg"])
        self.dec.load_state_dict(ckpt["generator"])
        self.dec.to(device).eval()
        # glide="next" slides the excitation pitch to the following frame's value, so it runs one frame behind
        self.late = int(self.dec.source.glide == "next")
        self.delay = DELAY + self.dec.lookahead + self.late  # frames between the newest input and the newest output
        assert self.enc.lookahead * 2 <= DELAY - 1 and self.pitch.lookahead + 4 <= DELAY + 1
        self.n = int(window * SR)
        self.ratio, self.offset, self.ceiling, self.keep, self.spk = 1.0, 0.0, 0.0, 1.0, None
        self.index, self.rate = None, 0.0  # the voice's index (causalvc.index) and its strength
        self.reset()

    def set_index(self, frames: torch.Tensor | None) -> None:
        """The voice's index frames [N, C] (None: no index)."""
        self.index = None if frames is None else frames.to(self.device, torch.float32)
        self.index_unit = None if frames is None else torch.nn.functional.normalize(self.index, dim=-1)

    def set_index_rate(self, rate: float) -> None:
        """How much of each content frame comes from the index (0 = none, 1 = all)."""
        self.rate = max(0.0, min(1.0, float(rate)))

    def set_voice(self, speaker: str, semitones: float = 0.0, offset_hz: float = 0.0, ceiling_hz: float = 0.0,
                  ceiling_keep: float = 1.0) -> None:
        """The voice to convert to and its pitch: the tracked pitch times 2^(semitones/12), plus ``offset_hz``,
        with only ``ceiling_keep`` of the rise above ``ceiling_hz`` (0 = no ceiling); see ``map_f0``."""
        self.spk = torch.tensor([self.speakers.index(speaker)], device=self.device)
        self.ratio, self.offset = 2.0 ** (semitones / 12.0), float(offset_hz)
        self.ceiling, self.keep = float(ceiling_hz), float(ceiling_keep)

    def reset(self) -> None:
        d = self.device
        self.buf = torch.zeros(self.n, device=d)
        self.content = torch.zeros(HIST, 768, device=d)
        self.f0 = torch.zeros(HIST, device=d)
        self.loud = torch.full((HIST,), -100.0, device=d)
        self.tone = torch.zeros(HIST, 2, device=d)
        self.exc = torch.zeros(self.dec.source.channels, HIST * FRAME, device=d)
        self.phase = torch.zeros(1, dtype=torch.float64, device=d)
        self.prev_f0 = torch.zeros(1, device=d)
        self.pending_f0 = torch.zeros(1, device=d)  # newest pitch frame, whose excitation waits for the next one
        self.level, self.level_pow, self.level_n, self.gain = -23.0, 0.0, 0, 1.0  # running speech level (dB, power), frames seen, gain

    LEVEL_BAND = 5.0  # dB above the target that still counts as an ordinary level and is left alone

    def normalise(self, block: torch.Tensor, m: int) -> torch.Tensor:
        """Slow automatic gain for microphones far from ordinary speech levels. The decoder
        is conditioned on loudness and was trained on speech at ordinary levels, so a very
        quiet microphone would otherwise be rendered as soft, dull speech.

        The speech level is the mean power of frames within 20 dB of the running level, so
        pauses and word tails do not drag it down. Nothing changes until 0.3 s of speech has
        been heard, input already between the target and LEVEL_BAND above it is left untouched,
        and the gain moves at most 10 dB per second."""
        db = loudness(block[None])[0]
        floor = max(self.level - 20.0, -60.0) if self.level_n else -50.0
        for v in db[db > floor].tolist():
            self.level_n += 1
            p = self.level_pow + (10 ** (v / 10) - self.level_pow) * max(1.0 / self.level_n, 1.0 / 300.0)
            self.level_pow, self.level = p, 10 * math.log10(p)
        want = 0.0
        if self.level_n >= 30:
            lo, hi = self.level_target, self.level_target + self.LEVEL_BAND
            want = min(max(lo - self.level, 0.0), 30.0) if self.level < lo else max(hi - self.level, -12.0) if self.level > hi else 0.0
        step = 0.1 * m  # dB this block may move
        now = 20 * math.log10(self.gain)
        gain = 10 ** ((now + min(max(want - now, -step), step)) / 20)
        ramp = torch.linspace(self.gain, gain, block.shape[0] + 1, device=block.device)[1:]
        self.gain = gain
        return (block * ramp).clamp(-1.0, 1.0)

    @torch.no_grad()
    def process(self, block: torch.Tensor) -> torch.Tensor:
        """[n] at 48 kHz (n a multiple of 960) -> [n], delayed by DELAY frames."""
        m = block.shape[0] // FRAME  # new 100 Hz frames
        assert block.shape[0] == m * FRAME and m % 2 == 0
        block = block.to(self.device)
        if self.level_target is not None:
            block = self.normalise(block, m)
        self.buf = torch.cat([self.buf[block.shape[0] :], block])
        total = self.n // FRAME
        frames = torch.arange(total - DELAY - m, total - DELAY, device=self.device)  # frames made final now

        wav16 = torchaudio.functional.resample(self.buf[None], SR, 16000)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.buf.is_cuda):
            steps = self.enc.forward(wav16).float()[0]  # step t is content frame t - lookahead
        content = steps[frames // 2 + self.enc.lookahead]
        if self.index is not None and self.rate > 0:
            from causalvc.index import lookup

            content = torch.lerp(content, lookup(content, self.index, self.index_unit), self.rate)
        sal = torch.sigmoid(self.pitch.forward(wav16).float())  # step t is pitch frame t - lookahead
        f0 = map_f0(self.pitch.decode(sal)[0, frames + self.pitch.lookahead], self.ratio, self.offset, self.ceiling, self.keep)
        loud = loudness(self.buf[None])[0, frames]

        if self.late:  # excitation for the frame before each new one, now that its following pitch is known
            exc = self.dec.source(torch.cat([self.pending_f0, f0[:-1]])[None], self.phase, nxt=f0[-1:])[0]
            self.pending_f0 = f0[-1:]
        else:
            exc = self.dec.source(f0[None], self.phase, self.prev_f0)[0]
        self.phase, self.prev_f0 = self.dec.source.last_phase, f0[-1:]
        self.content = torch.cat([self.content[m:], content])
        self.f0 = torch.cat([self.f0[m:], f0])
        self.loud = torch.cat([self.loud[m:], loud])
        if self.dec.tone is not None:
            self.tone = torch.cat([self.tone[m:], tone(self.buf[None])[0, frames]])
        self.exc = torch.cat([self.exc[:, m * FRAME :], exc], dim=1)

        dec = self.dec
        g = dec.speaker(self.spk)
        x = dec.content(self.content[None]) + dec.pitch(dec.pitch_bucket(self.f0[None]))
        x = x + dec.loud((self.loud[None, :, None] + 40.0) / 20.0)
        if dec.tone is not None:
            x = x + dec.tone(self.tone[None] / 10.0)
        x = x.transpose(1, 2)
        for b in dec.blocks:
            x = b(x, g)
        if "h" in dec.smooth_in:  # as Decoder.features does
            x = (x + torch.nn.functional.pad(x, (1, 0), mode="replicate")[..., :-1]) / 2
        k, a = m + GEN, dec.lookahead + self.late  # the last `a` frames still lack their future input or excitation
        end = x.shape[-1] - a
        e = self.late  # the excitation buffer runs this many frames behind the features
        y = dec.generate(x[..., end - k : end], g, self.exc[None, :, (end - k + e) * FRAME : (end + e) * FRAME])
        return y[0, 0, -m * FRAME :]
