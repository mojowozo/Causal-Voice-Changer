"""Live conversion: microphone in, converted voice out.

    python -m causalvc.live --voice runs/models/voice-NAME/voices/NAME-10000s.cvc
    python -m causalvc.live --voice ... --list-devices
    python -m causalvc.live --voice ... --in-file speech.wav --out-file converted.wav   (timing test, no audio devices)

The model converts each block inside the audio callback, so nothing is buffered
beyond what the sound card itself holds: delay is the model's fixed lookahead,
one block, the time to compute it, and the card's input and output buffers.
While running it prints how long blocks take to compute and how often the
callback was late. Works wherever PortAudio does (Linux, Windows, macOS).
"""

from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

SR = 48000
ROOT = Path(__file__).resolve().parent.parent


def load_voice(path: str, device: str):
    """A voice (.cvc file or .causalvc.json description) -> (streaming runtime, speaker name)."""
    from causalvc.fast import streaming

    from causalvc.voice import load

    spec = load(path)
    return streaming(spec["decoder"], spec["pitch"], device, spec["encoder"]), spec["speaker"]


class Live:
    """Block-by-block conversion with the controls a live user needs, and timing of every block."""

    def __init__(self, voice: str, device: str = "cuda:0", block_ms: int = 20, semitones: float = 0.0,
                 in_gain_db: float = 0.0, out_gain_db: float = 0.0, gate_db: float = -60.0, denoise: bool = False):
        assert block_ms % 20 == 0 and block_ms > 0, "the block must be a multiple of 20 ms"
        self.vc, self.speaker = load_voice(voice, device)
        self.device, self.block = device, SR * block_ms // 1000
        self.semitones, self.in_gain_db, self.out_gain_db, self.gate_db = semitones, in_gain_db, out_gain_db, gate_db
        self.vc.set_voice(self.speaker, semitones)
        self.vc.reset()
        # Input noise reduction (RNNoise); it adds 20 ms. `denoise` can be switched while running.
        from causalvc import rnnoise

        self.rnnoise = rnnoise.RNNoise() if rnnoise.available() else None
        self.denoise = denoise
        self.times: list[float] = []  # seconds of compute per block
        self.late = 0  # callbacks the sound card reported as too slow (audible gaps)
        self.lock = threading.Lock()
        for _ in range(8):  # the first blocks are slow while the GPU libraries settle on their algorithms
            self.convert(np.zeros(self.block, dtype=np.float32), timed=False)
        self.vc.reset()

    @property
    def model_delay_ms(self) -> float:
        return self.vc.delay * 10.0 + (20.0 if self.denoise and self.rnnoise is not None else 0.0)

    def set_pitch(self, semitones: float) -> None:
        with self.lock:
            self.semitones = semitones
            self.vc.set_voice(self.speaker, semitones)

    def convert(self, block: np.ndarray, timed: bool = True) -> np.ndarray:
        """One block of mono float32 at 48 kHz in, the same number of samples out."""
        t0 = time.perf_counter()
        x = torch.from_numpy(block) * 10 ** (self.in_gain_db / 20)
        level = 20 * math.log10(float(x.pow(2).mean().sqrt()) + 1e-7)
        if level < self.gate_db:
            x = torch.zeros_like(x)  # below the gate: feed silence, so the model's own state keeps running
        if self.denoise and self.rnnoise is not None:
            x = torch.from_numpy(self.rnnoise.process(x.clamp(-1.0, 1.0).numpy()))
        with self.lock:
            y = self.vc.process(x)
        out = (y * 10 ** (self.out_gain_db / 20)).clamp(-1.0, 1.0).cpu().numpy()  # .cpu() waits for the GPU
        if timed:
            self.times.append(time.perf_counter() - t0)
        return out

    def stats(self, last: int = 0) -> dict[str, float]:
        t = np.array(self.times[-last:] if last else self.times) * 1000
        if not len(t):
            return {}
        block_ms = self.block / SR * 1000
        return {"blocks": len(t), "median_ms": float(np.median(t)), "p99_ms": float(np.percentile(t, 99)), "max_ms": float(t.max()),
                "over_block_pct": float((t > block_ms).mean() * 100), "late": self.late}

    def stream(self, in_device=None, out_device=None, latency="low"):
        """A running duplex audio stream. Keep the returned object; call .stop() and .close() when done."""
        import sounddevice as sd

        def callback(indata, outdata, frames, when, status):
            if status.input_overflow or status.output_underflow:
                self.late += 1
            outdata[:, 0] = self.convert(indata[:, 0].copy())

        extra = None
        if sys.platform == "win32":
            # WASAPI shared mode only opens at the device's own rate unless asked to convert; the model needs 48 kHz.
            try:
                extra = sd.WasapiSettings(auto_convert=True)
            except TypeError:  # older sounddevice
                extra = None
        try:
            s = sd.Stream(samplerate=SR, blocksize=self.block, device=(in_device, out_device), channels=1,
                          dtype="float32", latency=latency, callback=callback, extra_settings=extra)
        except sd.PortAudioError:
            if extra is None:
                raise
            # not WASAPI devices (MME, DirectSound): those convert by themselves
            s = sd.Stream(samplerate=SR, blocksize=self.block, device=(in_device, out_device), channels=1,
                          dtype="float32", latency=latency, callback=callback)
        s.start()
        return s


def describe(live: Live, stream=None, last: int = 0) -> str:
    s = live.stats(last)
    if not s:
        return "no blocks yet"
    block_ms = live.block / SR * 1000
    text = (f"compute per {block_ms:.0f} ms block: median {s['median_ms']:.1f} ms, 99% under {s['p99_ms']:.1f} ms, worst {s['max_ms']:.1f} ms; "
            f"{s['over_block_pct']:.1f}% of blocks took longer than a block; late callbacks {s['late']}")
    if stream is not None:
        i, o = (x * 1000 for x in stream.latency)
        total = live.model_delay_ms + block_ms + s["median_ms"] + i + o
        text += (f"\n  delay estimate: model {live.model_delay_ms:.0f} + block {block_ms:.0f} + compute {s['median_ms']:.0f} "
                 f"+ card in {i:.0f} + card out {o:.0f} = {total:.0f} ms (the card figures are what it reports)")
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", help="a .cvc voice file (or an installed voice's .causalvc.json)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--block", type=int, default=20, help="block length in ms (a multiple of 20)")
    ap.add_argument("--pitch", type=float, default=0.0, help="pitch shift in semitones")
    ap.add_argument("--in-gain", type=float, default=0.0, help="dB")
    ap.add_argument("--out-gain", type=float, default=0.0, help="dB")
    ap.add_argument("--gate", type=float, default=-60.0, help="input blocks quieter than this (dB) are treated as silence")
    ap.add_argument("--denoise", action="store_true", help="input noise reduction (RNNoise); adds 20 ms")
    ap.add_argument("--in-device", help="input device name or number (default: the system's)")
    ap.add_argument("--out-device", help="output device name or number (default: the system's)")
    ap.add_argument("--latency", default="low", help="sound card buffering to ask for: low, high, or seconds")
    ap.add_argument("--seconds", type=float, default=0.0, help="stop after this long (default: until Ctrl-C)")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--in-file", help="convert this file block by block instead of opening audio devices")
    ap.add_argument("--out-file")
    a = ap.parse_args()

    if a.list_devices:
        import sounddevice as sd

        print(sd.query_devices())
        return
    if not a.voice:
        sys.exit("--voice is required")
    live = Live(a.voice, a.device, a.block, a.pitch, a.in_gain, a.out_gain, a.gate, a.denoise)
    print(f"voice {live.speaker}; model delay {live.model_delay_ms:.0f} ms + {a.block} ms block", flush=True)

    if a.in_file:
        import soundfile as sf
        import torchaudio

        wav, sr = sf.read(a.in_file, dtype="float32", always_2d=True)
        x = torch.from_numpy(wav[:, 0])
        x = (torchaudio.functional.resample(x, sr, SR) if sr != SR else x).numpy()
        out = [live.convert(x[i : i + live.block]) for i in range(0, len(x) - live.block + 1, live.block)]
        if a.out_file:
            sf.write(a.out_file, np.concatenate(out), SR, subtype="PCM_16")
        print(describe(live), flush=True)
        return

    num = lambda v: int(v) if v is not None and v.isdigit() else v  # noqa: E731
    latency = a.latency if a.latency in ("low", "high") else float(a.latency)
    stream = live.stream(num(a.in_device), num(a.out_device), latency)
    print("running; Ctrl-C to stop", flush=True)
    start, seen = time.time(), 0
    try:
        while not a.seconds or time.time() - start < a.seconds:
            time.sleep(2.0)
            n = len(live.times)
            print(describe(live, stream, n - seen), flush=True)
            seen = n
    except KeyboardInterrupt:
        pass
    stream.stop()
    stream.close()
    print("overall: " + describe(live, stream), flush=True)


if __name__ == "__main__":
    main()
