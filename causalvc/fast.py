"""Block-by-block conversion that computes only what is new.

`causalvc.stream.StreamingVC` re-runs the encoder and pitch tracker over the last
four seconds for every block and regenerates a third of a second of waveform to
emit 20 ms. Every network here only looks backward, so all of that older work is
already final. This runtime keeps, for each layer, the little history the layer
needs (a few frames for a convolution, the attention window for the encoder) and
runs each layer on the new frames alone. Same weights, same output, a fraction
of the work.

Same interface as StreamingVC. Models with decoder lookahead or the diagnostic
conditioning options are not handled here; `streaming()` falls back for those.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torchaudio

from causalvc.decoder.model import FRAME, SR, Decoder, loudness
from causalvc.infer import load_encoder
from causalvc.pitch_causal import load_pitch
from causalvc.stream import DELAY

TAIL = 3840  # 48 kHz samples of input kept between blocks (80 ms): covers every front end's history


def supported(dec: Decoder) -> bool:
    return dec.lookahead == 0 and not dec.smooth_in and dec.mel is None and dec.exc == "full"


class FastStreamingVC:
    def __init__(self, decoder: str, pitch: str, device: str = "cuda:0", encoder: str | None = None, level: float | None = None):
        assert level is None, "the automatic input gain lives in StreamingVC only"
        ckpt = torch.load(decoder, map_location="cpu")
        self.device, self.speakers = device, ckpt["speakers"]
        self.enc = load_encoder(encoder or ckpt["encoder"], device)
        self.pitch = load_pitch(pitch, device)
        self.dec = Decoder(len(self.speakers), **ckpt["cfg"])
        self.dec.load_state_dict(ckpt["generator"])
        self.dec.to(device).eval().requires_grad_(False)
        assert supported(self.dec), "this decoder needs causalvc.stream.StreamingVC"
        # Weight normalisation recomputes every kernel on every call; the weights are fixed now, so store them plain.
        for mod in self.dec.modules():
            if torch.nn.utils.parametrize.is_parametrized(mod, "weight"):
                torch.nn.utils.parametrize.remove_parametrizations(mod, "weight", leave_parametrized=True)
        # Rotary position factors for every length the attention cache can have, computed once.
        e = self.enc
        d = e.blocks[0].qkv.out_features // 3 // e.blocks[0].heads // 2
        freq = torch.exp(-torch.log(torch.tensor(10000.0)) * torch.arange(d, device=device) / d)
        ang = torch.arange(e.window + 64, device=device)[:, None].float() * freq[None]
        self.rot = (ang.cos(), ang.sin())
        self.late = int(self.dec.source.glide == "next")
        self.delay = DELAY + self.late
        assert self.enc.lookahead * 2 <= DELAY - 1 and self.pitch.lookahead + 4 <= DELAY + 1
        self.ratio, self.spk = 1.0, None
        self.reset()

    def set_voice(self, speaker: str, semitones: float = 0.0) -> None:
        self.spk = torch.tensor([self.speakers.index(speaker)], device=self.device)
        self.ratio = 2.0 ** (semitones / 12.0)

    def reset(self) -> None:
        d = self.device
        self.tail = torch.zeros(TAIL, device=d)  # newest input, 48 kHz
        self.st: dict = {}  # per-layer history, created on first use
        self.steps: list[torch.Tensor] = []  # newest encoder outputs [768]; the last is the newest step
        self.phase = torch.zeros(1, dtype=torch.float64, device=d)
        self.prev_f0 = torch.zeros(1, device=d)
        self.pending_f0 = torch.zeros(1, device=d)
        self.pending_h = None  # features of the newest frame, waiting for its excitation (glide="next")
        self.frames = 0  # 100 Hz frames of input received so far

    # ---- history helpers -------------------------------------------------

    def _hist(self, key, x: torch.Tensor, n: int) -> torch.Tensor:
        """x [..., T] preceded by the last n samples seen under this key (zeros at the start); remembers the new tail."""
        old = self.st.get(key)
        if old is None:
            old = x.new_zeros(*x.shape[:-1], n)
        z = torch.cat([old, x], dim=-1)
        self.st[key] = z[..., z.shape[-1] - n :]
        return z

    def _cconv(self, m, x: torch.Tensor, key) -> torch.Tensor:
        """A causal convolution (decoder.model.CConv) on new frames only."""
        return m.conv(self._hist(key, x, m.pad)) if m.pad else m.conv(x)

    # ---- content encoder -------------------------------------------------

    def _encode(self, wav16: torch.Tensor, n: int) -> torch.Tensor:
        """The last 240 + 320 n samples at 16 kHz -> n new encoder steps [n, 768]."""
        e = self.enc
        m = torch.log(e.mel(wav16[None]).clamp_min(1e-5))  # [1, mels, 2n]
        h = F.gelu(e.conv1(self._hist("e.mel", m, 2)))
        x = F.gelu(e.conv2(self._hist("e.h", h, 2))).transpose(1, 2)  # [1, n, dim]
        for i, blk in enumerate(e.blocks):
            b, t, c = x.shape
            q, k, v = blk.qkv(blk.n1(x)).view(b, t, 3, blk.heads, c // blk.heads).permute(2, 0, 3, 1, 4)
            if ("e.k", i) in self.st:
                k, v = torch.cat([self.st["e.k", i], k], dim=2), torch.cat([self.st["e.v", i], v], dim=2)
            total = k.shape[2]
            # keep what the next step may attend to: itself plus `window` frames back means `window` cached frames
            self.st["e.k", i], self.st["e.v", i] = k[:, :, -e.window :], v[:, :, -e.window :]
            if t == 1:  # one new step sees every cached key: the cache is exactly its window
                a = F.scaled_dot_product_attention(self._rot(q, total - 1, total), self._rot(k, 0, total), v)
            else:
                pos = torch.arange(total, device=x.device)
                rel = pos[-t:, None] - pos[None, :]
                a = F.scaled_dot_product_attention(self._rot(q, total - t, total), self._rot(k, 0, total), v, attn_mask=(rel >= 0) & (rel <= e.window))
            x = x + blk.proj(a.transpose(1, 2).reshape(b, t, c))
            hh = self._hist(("e.c", i), blk.n2(x).transpose(1, 2), blk.kernel - 1)
            x = x + blk.conv_out(F.gelu(blk.conv(hh).transpose(1, 2)))
            x = x + blk.ff(blk.n3(x))
        return e.head(e.norm(x))[0]

    def _rot(self, x: torch.Tensor, a: int, b: int) -> torch.Tensor:
        """encoder.model.rotary for positions a..b-1, from the stored factors."""
        cos, sin = self.rot[0][a:b], self.rot[1][a:b]
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    # ---- pitch tracker ---------------------------------------------------

    def _pitch(self, wav16: torch.Tensor, m: int) -> torch.Tensor:
        """The last 1120 + 160 m samples at 16 kHz -> F0 of m new frames [m] (before the pitch shift)."""
        p = self.pitch
        mel = torch.log(p.mel(wav16[None]).clamp_min(1e-5))[..., 4 : 4 + m]  # frames with a full window on both sides
        x = p.inp(self._hist("p.mel", mel, 2))
        for i, b in enumerate(p.blocks):
            y = b.dw(self._hist(("p.b", i), x, b.pad)).transpose(1, 2)
            x = x + b.fc2(F.gelu(b.fc1(b.norm(y)))).transpose(1, 2)
        x = x.transpose(1, 2)
        g, self.st["p.gru"] = p.gru(x, self.st.get("p.gru"))
        return p.decode(torch.sigmoid(p.head(torch.cat([x, g], dim=-1))))[0]

    # ---- decoder ---------------------------------------------------------

    def _features(self, content: torch.Tensor, f0: torch.Tensor, loud: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """content [m, 768], f0 [m], loud [m] for the new frames -> hidden [1, D, m]."""
        d = self.dec
        x = d.content(content[None]) + d.pitch(d.pitch_bucket(f0[None])) + d.loud((loud[None, :, None] + 40.0) / 20.0)
        x = x.transpose(1, 2)
        for i, b in enumerate(d.blocks):
            y = self._cconv(b.dw, x, ("d.b", i)).transpose(1, 2) + b.cond(g)[:, None]
            x = x + b.fc2(F.gelu(b.fc1(b.norm(y)))).transpose(1, 2)
        return x

    def _generate(self, h: torch.Tensor, g: torch.Tensor, exc: torch.Tensor) -> torch.Tensor:
        """hidden [1, D, m] and excitation [1, C, m * FRAME] for the same frames -> waveform [m * FRAME]."""
        d = self.dec
        x = self._cconv(d.pre, h, "g.pre") + d.pre_cond(g)[..., None]
        for i, (up, src, res) in enumerate(zip(d.ups, d.src, d.res)):
            x = F.leaky_relu(x, 0.1)
            z = self._hist(("g.up", i), x, 1)  # a transposed convolution's output hop depends on this frame and the one before
            x = up.conv(z)[..., up.stride : z.shape[-1] * up.stride] + src(exc)
            out = 0
            for j, r in enumerate(res):
                y = x
                for k, (c1, c2) in enumerate(zip(r.c1, r.c2)):
                    t = self._cconv(c1, F.leaky_relu(y, 0.1), ("g.r", i, j, k, 1))
                    y = y + self._cconv(c2, F.leaky_relu(t, 0.1), ("g.r", i, j, k, 2))
                out = out + y
            x = out / len(res)
        return torch.tanh(self._cconv(d.post, F.leaky_relu(x, 0.1), "g.post"))[0, 0]

    # ---- one block -------------------------------------------------------

    @torch.inference_mode()
    def process(self, block: torch.Tensor) -> torch.Tensor:
        """[n] at 48 kHz (n a multiple of 960) -> [n], delayed by `delay` frames."""
        m = block.shape[0] // FRAME
        assert block.shape[0] == m * FRAME and m % 2 == 0
        buf = torch.cat([self.tail, block.to(self.device)])
        self.tail = buf[-TAIL:]
        self.frames += m

        wav16 = torchaudio.functional.resample(buf[None], SR, 16000)[0]
        for step in self._encode(wav16[-(240 + 160 * m) :], m // 2):
            self.steps.append(step)
        self.steps = self.steps[-(m // 2 + 2) :]
        # Frame f is final now (f = frames - DELAY - m .. frames - DELAY - 1); it uses content frame f // 2,
        # which is encoder step f // 2 + lookahead. The newest step is frames // 2 - 1.
        newest = self.frames // 2 - 1
        first = self.frames - DELAY - m
        zero = self.steps[0].new_zeros(self.steps[0].shape)
        content = []
        for f in range(first, first + m):
            back = newest - (f // 2 + self.enc.lookahead)  # how many steps before the newest
            content.append(self.steps[-1 - back] if f >= 0 and back < len(self.steps) else zero)
        content = torch.stack(content)

        f0 = self._pitch(wav16[-(1120 + 160 * m) :], m) * self.ratio
        if first < 0:  # before any real frame is final the old runtime's history is silence too
            f0 = torch.where(torch.arange(first, first + m, device=f0.device) >= 0, f0, torch.zeros_like(f0))
        loud = loudness(buf[None, buf.shape[0] - (DELAY + m) * FRAME : buf.shape[0] - DELAY * FRAME])[0]

        g = self.dec.speaker(self.spk)
        h = self._features(content, f0, loud, g)
        if self.late:  # the excitation of a frame needs the next frame's pitch, so the waveform runs one frame behind
            exc = self.dec.source(torch.cat([self.pending_f0, f0[:-1]])[None], self.phase, nxt=f0[-1:])
            self.pending_f0 = f0[-1:]
            held = h.new_zeros(1, h.shape[1], 1) if self.pending_h is None else self.pending_h
            h, self.pending_h = torch.cat([held, h[..., :-1]], dim=-1), h[..., -1:]
        else:
            exc = self.dec.source(f0[None], self.phase, self.prev_f0)
        self.phase, self.prev_f0 = self.dec.source.last_phase, f0[-1:]
        return self._generate(h, g, exc)


def streaming(decoder: str, pitch: str, device: str = "cuda:0", encoder: str | None = None, level: float | None = None):
    """The fastest runtime that handles this decoder."""
    cfg = torch.load(decoder, map_location="cpu")["cfg"]
    plain = not cfg.get("lookahead") and not cfg.get("smooth_in") and not cfg.get("mel_cond") and cfg.get("exc", "full") == "full"
    if plain and level is None:
        return FastStreamingVC(decoder, pitch, device, encoder)
    from causalvc.stream import StreamingVC

    return StreamingVC(decoder, pitch, device, encoder, level=level)
