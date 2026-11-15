"""Causal source-filter decoder: content + pitch + loudness + speaker -> 48 kHz audio.

Every layer only looks backwards, so the waveform it produces for a frame is
final as soon as that frame's features exist: streamed output equals whole-file
output, whatever the block size. The decoder itself adds no lookahead.

Frame clock is 100 Hz (480 samples). Content arrives at 50 Hz and is repeated.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

SR = 48000
FRAME = 480  # samples per 100 Hz frame


class CConv(nn.Module):
    """Causal 1-D convolution (pads on the left only)."""

    def __init__(self, cin: int, cout: int, k: int, dilation: int = 1, groups: int = 1, norm: bool = True, ahead: int = 0):
        super().__init__()
        self.pad, self.ahead = (k - 1) * dilation, ahead  # `ahead` extra taps look at future steps
        conv = nn.Conv1d(cin, cout, k + ahead, dilation=dilation, groups=groups)
        self.conv = weight_norm(conv) if norm else conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.pad, self.ahead)))


class CUp(nn.Module):
    """Causal transposed convolution: output hop t depends on frames t-1 and t."""

    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.stride = stride
        self.conv = weight_norm(nn.ConvTranspose1d(cin, cout, 2 * stride, stride))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)[..., : x.shape[-1] * self.stride]


class ResBlock(nn.Module):
    def __init__(self, ch: int, k: int, dilations: tuple[int, ...] = (1, 3, 5)):
        super().__init__()
        self.c1 = nn.ModuleList(CConv(ch, ch, k, d) for d in dilations)
        self.c2 = nn.ModuleList(CConv(ch, ch, k, 1) for _ in dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.c1, self.c2):
            x = x + c2(F.leaky_relu(c1(F.leaky_relu(x, 0.1)), 0.1))
        return x


class NextBlock(nn.Module):
    """Causal ConvNeXt block with a speaker-dependent shift."""

    def __init__(self, dim: int, spk: int, k: int = 7, ahead: int = 0):
        super().__init__()
        self.dw = CConv(dim, dim, k, groups=dim, norm=False, ahead=ahead)
        self.norm = nn.LayerNorm(dim)
        self.fc1, self.fc2 = nn.Linear(dim, 3 * dim), nn.Linear(3 * dim, dim)
        self.cond = nn.Linear(spk, dim)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:  # x [B, C, T], g [B, S]
        y = self.dw(x).transpose(1, 2) + self.cond(g)[:, None]
        y = self.fc2(F.gelu(self.fc1(self.norm(y))))
        return x + y.transpose(1, 2)


class Source(nn.Module):
    """Harmonic-plus-noise excitation at the sample rate, from frame-rate F0.

    F0 is held for the frame it belongs to; phase is accumulated, so pitch
    steps do not click. Returns [B, harmonics + 1 + bands, N]: the first
    harmonics as separate sines, a voicing flag, then one channel per frequency
    band holding the sum of every harmonic that falls in it (up to Nyquist), so
    the decoder is handed harmonic structure across the whole spectrum.
    """

    EDGES = (0.0, 4000.0, 8000.0, 16000.0, SR / 2)

    def __init__(self, harmonics: int = 8, amp: float = 0.1, noise: float = 0.003, bands: int = 0, glide: bool | str = False, with_f0: bool = False):
        super().__init__()
        assert bands in (0, len(self.EDGES) - 1)
        self.h, self.amp, self.noise, self.bands, self.glide = harmonics, amp, noise, bands, glide
        self.last_phase = None  # phase after the last sample, for the next streamed call
        self.n_in = harmonics + 1 + bands  # channels the decoder's excitation convs read
        self.with_f0 = with_f0  # append per-sample F0 as a last channel, for per-stage band-limiting
        self.channels = self.n_in + int(with_f0)

    @staticmethod
    def _dirichlet(k: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """sum_{j=1..k} sin(j * theta), in closed form."""
        half = torch.sin(theta / 2)
        ok = half.abs() > 1e-7
        return torch.where(ok, torch.sin(k * theta / 2) * torch.sin((k + 1) * theta / 2) / torch.where(ok, half, 1.0), 0.0)

    def forward(self, f0: torch.Tensor, phase0: torch.Tensor | None = None, prev: torch.Tensor | None = None, nxt: torch.Tensor | None = None) -> torch.Tensor:
        """f0 [B, T]; phase0 [B] and prev [B] (the F0 of the frame before) carry a stream across calls;
        nxt [B] is the F0 of the frame after the last one, for glide="next"."""
        f = f0.repeat_interleave(FRAME, dim=1).double()  # [B, N]
        if self.glide == "next":
            # A pitch frame is the pitch at the *start* of its 10 ms (measured on both trackers), so holding
            # it for the frame plays every pitch 5 ms late and in steps. Slide to the next frame's value instead.
            after = torch.cat([f0[:, 1:], f0[:, -1:] if nxt is None else nxt[:, None]], dim=1)
            ramp = torch.arange(FRAME, device=f0.device, dtype=torch.float64) / FRAME
            slide = (f0[..., None] + (after - f0)[..., None] * ramp).flatten(1)
            both = ((after > 0) & (f0 > 0)).repeat_interleave(FRAME, dim=1)
            f = torch.where(both, slide, f)
        elif self.glide:  # slide from the previous frame's pitch (this is 10 ms late; kept for old checkpoints)
            before = torch.cat([f0[:, :1] if prev is None else prev[:, None], f0[:, :-1]], dim=1)
            ramp = torch.arange(1, FRAME + 1, device=f0.device, dtype=torch.float64) / FRAME
            slide = (before[..., None] + (f0 - before)[..., None] * ramp).flatten(1)
            both = ((before > 0) & (f0 > 0)).repeat_interleave(FRAME, dim=1)
            f = torch.where(both, slide, f)
        phase = torch.cumsum(f / SR, dim=1)
        if phase0 is not None:
            phase = phase + phase0[:, None]
        self.last_phase = phase[:, -1] % 1.0
        k = torch.arange(1, self.h + 1, device=f0.device, dtype=torch.float64)[None, :, None]
        sines = torch.sin(2 * math.pi * ((phase[:, None] * k) % 1.0)).float()
        voiced = (f > 0).float()[:, None]
        std = voiced * self.noise + (1 - voiced) * self.amp / 3
        out = [sines * (f.float()[:, None] * k.float() < SR / 2) * voiced * self.amp]
        if self.bands:
            theta = 2 * math.pi * (phase % 1.0)
            fs = f.clamp_min(1.0)
            top = [torch.floor(min(e, SR / 2 - 1.0) / fs) for e in self.EDGES]  # harmonics at or below each edge
            for lo, hi in zip(top[:-1], top[1:]):
                band = (self._dirichlet(hi, theta) - self._dirichlet(lo, theta)) / (hi - lo).clamp_min(1.0).sqrt()
                out.append(band.float()[:, None] * voiced * self.amp)
        x = torch.cat(out, dim=1)
        x = x + std * torch.randn_like(x)
        out = [x[:, : self.h], voiced, x[:, self.h :]]
        return torch.cat(out + ([f.float()[:, None]] if self.with_f0 else []), dim=1)


class Decoder(nn.Module):
    def __init__(
        self,
        n_speakers: int,
        content_dim: int = 768,
        dim: int = 256,
        spk_dim: int = 256,
        blocks: int = 6,
        channels: int = 384,
        rates: tuple[int, ...] = (10, 6, 4, 2),
        kernels: tuple[int, ...] = (3, 7, 11),
        harmonics: int = 8,
        bands: int = 0,
        glide: bool | str = False,
        lookahead: int = 0,
        mel_cond: bool = False,
        exc: str = "full",
        smooth_in: str = "",
        tone: int = 0,
    ):
        super().__init__()
        assert math.prod(rates) == FRAME
        self.speaker = nn.Embedding(n_speakers, spk_dim)
        self.content = nn.Linear(content_dim, dim)
        self.pitch = nn.Embedding(256, dim)
        self.loud = nn.Linear(1, dim)
        # Optional coarse description of the input's tone per frame (see `tone` below): what content
        # features say least about in non-speech sounds such as laughter. Starts as a no-op.
        self.tone = nn.Linear(tone, dim) if tone else None
        if tone:
            nn.init.zeros_(self.tone.weight), nn.init.zeros_(self.tone.bias)
        # Diagnostic only: condition on the target's own mel spectrogram (copy-synthesis ceiling).
        self.mel = nn.Linear(128, dim) if mel_cond else None
        if mel_cond:
            nn.init.zeros_(self.mel.weight), nn.init.zeros_(self.mel.bias)
        self.lookahead = lookahead  # 100 Hz frames of future input, one per block from the last
        self.blocks = nn.ModuleList(NextBlock(dim, spk_dim, ahead=int(i >= blocks - lookahead)) for i in range(blocks))
        assert exc in ("full", "bandlimit", "fundamental")
        self.exc = exc
        # Conditioning without frame-rate steps: "c" interpolates the 50 Hz content,
        # "p" blends neighbouring pitch buckets, "h" averages each hidden frame with the one before.
        self.smooth_in = smooth_in
        self.source = Source(harmonics, bands=bands, glide=glide, with_f0=exc != "full")
        self.stage_rates = []
        self.pre = CConv(dim, channels, 7)
        self.pre_cond = nn.Linear(spk_dim, channels)
        self.ups, self.src, self.res = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        ch, remaining = channels, FRAME
        for r in rates:
            self.ups.append(CUp(ch, ch // 2, r))
            ch, remaining = ch // 2, remaining // r
            self.stage_rates.append(SR // remaining)
            # Each output step sees only the excitation of its own hop.
            self.src.append(nn.Conv1d(self.source.n_in, ch, remaining, remaining))
            self.res.append(nn.ModuleList(ResBlock(ch, k) for k in kernels))
        self.post = CConv(ch, 1, 7)

    @staticmethod
    def pitch_position(f0: torch.Tensor) -> torch.Tensor:
        """Fractional bucket position (1..255) on the same scale as pitch_bucket."""
        mel = 1127.0 * torch.log1p(f0.clamp_min(1.0) / 700.0)
        lo, hi = 1127.0 * math.log1p(50 / 700), 1127.0 * math.log1p(1100 / 700)
        return ((mel - lo) / (hi - lo) * 254 + 1).clamp(1, 255)

    @staticmethod
    def pitch_bucket(f0: torch.Tensor) -> torch.Tensor:
        mel = 1127.0 * torch.log1p(f0 / 700.0)
        lo, hi = 1127.0 * math.log1p(50 / 700), 1127.0 * math.log1p(1100 / 700)
        b = ((mel - lo) / (hi - lo) * 254 + 1).round().clamp(1, 255).long()
        return torch.where(f0 > 0, b, torch.zeros_like(b))

    def features(self, content: torch.Tensor, f0: torch.Tensor, loud: torch.Tensor, spk: torch.Tensor, mel: torch.Tensor | None = None,
                 tone: torch.Tensor | None = None):
        """content [B, T/2, C] at 50 Hz; f0 (Hz, 0 unvoiced) and loud (dB) [B, T]
        at 100 Hz; spk [B]. Returns hidden [B, D, T] and speaker vector."""
        g = self.speaker(spk)
        t = min(content.shape[1] * 2, f0.shape[1], loud.shape[1])
        x = self.content(content)
        if "c" in self.smooth_in:  # frame 2i is content i, frame 2i+1 the mean of i and i+1 would need the future: use i-1..i instead
            x = torch.stack([(x + torch.cat([x[:, :1], x[:, :-1]], 1)) / 2, x], dim=2).flatten(1, 2)[:, :t]
        else:
            x = x.repeat_interleave(2, dim=1)[:, :t]
        if "p" in self.smooth_in:
            pos = self.pitch_position(f0[:, :t])
            lo = pos.floor().long().clamp(1, 254)
            w = (pos - lo).clamp(0, 1)[..., None]
            pe = self.pitch(lo) * (1 - w) + self.pitch(lo + 1) * w
            pe = torch.where((f0[:, :t] > 0)[..., None], pe, self.pitch(torch.zeros_like(lo)))
        else:
            pe = self.pitch(self.pitch_bucket(f0[:, :t]))
        x = x + pe + self.loud((loud[:, :t, None] + 40.0) / 20.0)
        if self.mel is not None:
            x = x + self.mel(mel[:, :t])
        if self.tone is not None and tone is not None:
            x = x + self.tone(tone[:, :t] / 10.0)
        x = x.transpose(1, 2)
        for block in self.blocks:
            x = block(x, g)
        if "h" in self.smooth_in:
            x = (x + F.pad(x, (1, 0), mode="replicate")[..., :-1]) / 2
        return x, g

    def generate(self, h: torch.Tensor, g: torch.Tensor, excitation: torch.Tensor) -> torch.Tensor:
        """h [B, D, T], excitation [B, H+1, T * FRAME] -> waveform [B, 1, T * FRAME]."""
        x = self.pre(h) + self.pre_cond(g)[..., None]
        f = None
        if self.exc != "full":
            excitation, f = excitation[:, :-1], excitation[:, -1:]
        for up, src, res, rate in zip(self.ups, self.src, self.res, self.stage_rates):
            x = up(F.leaky_relu(x, 0.1)) + src(excitation if f is None else self.limit(excitation, f, rate))
            x = sum(r(x) for r in res) / len(res)
        return torch.tanh(self.post(F.leaky_relu(x, 0.1)))

    def limit(self, e: torch.Tensor, f: torch.Tensor, rate: int) -> torch.Tensor:
        """Drop, where voiced, the excitation partials a stage running at `rate` cannot
        represent: its input conv decimates without a filter, so they would fold back
        as inharmonic components. "fundamental" keeps only the first partial."""
        h, nb = self.source.h, self.source.bands
        k = torch.arange(1, h + 1, device=e.device, dtype=e.dtype)[None, :, None]
        keep = (f * k < 0.45 * rate) if self.exc == "bandlimit" else (k == 1) & (f < 0.45 * rate)
        keep = keep | (f <= 0)  # unvoiced: noise in every channel, as before
        parts = [e[:, :h] * keep, e[:, h : h + 1]]
        if nb:
            ok = torch.tensor([hi <= rate / 2 and self.exc == "bandlimit" for hi in Source.EDGES[1:]], device=e.device)[None, :, None]
            parts.append(e[:, h + 1 :] * (ok | (f <= 0)))
        return torch.cat(parts, dim=1)

    def forward(self, content, f0, loud, spk) -> torch.Tensor:
        h, g = self.features(content, f0, loud, spk)
        return self.generate(h, g, self.source(f0[:, : h.shape[-1]]))


def loudness(wav: torch.Tensor) -> torch.Tensor:
    """[B, N] at 48 kHz -> dB RMS of each 10 ms frame, [B, N // FRAME]."""
    n = wav.shape[1] // FRAME
    frames = wav[:, : n * FRAME].reshape(wav.shape[0], n, FRAME)
    return 20 * torch.log10(frames.pow(2).mean(-1).sqrt() + 1e-5)


TONE_BANDS = ((2000, 8000, 250, 2000), (700, 2000, 250, 700))  # brightness, vowel openness


def tone(wav: torch.Tensor) -> torch.Tensor:
    """[B, N] at 48 kHz -> [B, N // FRAME, 2] in dB: brightness (2-8 kHz against 0.25-2 kHz) and
    openness (0.7-2 kHz against 0.25-0.7 kHz) of each 10 ms frame, from that frame and the one before."""
    n = wav.shape[1] // FRAME
    x = F.pad(wav[:, : n * FRAME], (FRAME, 0)).unfold(1, 2 * FRAME, FRAME)  # [B, n, 960]
    S = torch.fft.rfft(x * torch.hann_window(2 * FRAME, device=wav.device)).abs().pow(2)
    f = torch.arange(S.shape[-1], device=wav.device) * SR / (2 * FRAME)
    e = lambda lo, hi: S[..., (f >= lo) & (f < hi)].sum(-1) + 1e-4  # noqa: E731  (the floor keeps silence near 0 dB)
    return torch.stack([10 * torch.log10(e(a, b) / e(c, d)) for a, b, c, d in TONE_BANDS], dim=-1)


class CausalMel(nn.Module):
    """Log-mel at 100 Hz where frame t only uses samples up to the end of frame t."""

    def __init__(self):
        super().__init__()
        import torchaudio

        self.m = torchaudio.transforms.MelSpectrogram(SR, n_fft=1024, hop_length=FRAME, n_mels=128, center=False, power=1.0)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:  # [B, N] -> [B, T, 128]
        return torch.log(self.m(F.pad(wav, (1024 - FRAME, 0))).clamp_min(1e-5)).transpose(1, 2)
