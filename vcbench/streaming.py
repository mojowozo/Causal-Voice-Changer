"""System interface and the block-by-block streaming driver.

Dependency-light on purpose: a system adapter runs inside whatever environment
its model needs, and only has to import this module and `vcbench.audio`.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Voice:
    """A target voice: the audio a system may build its model from."""

    id: str
    train_wavs: list[str]
    workdir: Path  # per-system, per-voice scratch space for trained artifacts


class VCSystem(ABC):
    """A voice conversion system driven like a realtime audio callback.

    Blocks go in and come out at `io_sr`, with equal length. Any resampling,
    context buffering or crossfading is the adapter's job, exactly as it would
    be in a live application.
    """

    name: str = "base"
    io_sr: int = 48000

    def build_voices(self, voices: list[Voice]) -> None:
        """Hook for systems that train all target voices as one model."""

    def build_voice(self, voice: Voice) -> None:
        """Train or enrol the target voice. Must be idempotent (skip if done)."""

    @abstractmethod
    def load_voice(self, voice: Voice) -> None:
        """Make `voice` the active target."""

    def set_pitch_shift(self, semitones: int) -> None:
        """Static transposition for the next utterance, as a user would dial in
        for their own voice (median-F0 ratio of target to source speaker)."""

    def set_block_size(self, n_samples: int) -> None:
        """Called before streaming with the block length at `io_sr`."""

    def reset(self) -> None:
        """Clear all streaming state (start of a new utterance)."""

    @abstractmethod
    def process_block(self, block: np.ndarray) -> np.ndarray:
        """float32 mono in, float32 mono out, same length."""

    def convert_offline(self, wav: np.ndarray) -> np.ndarray | None:
        """Whole-utterance conversion with unlimited context, if supported."""
        return None

    def reported_latency_ms(self) -> float | None:
        """Algorithmic latency the system claims, excluding block buffering."""
        return None

    def sync(self) -> None:
        """Block until pending GPU work is finished (for honest timing)."""


@dataclass
class StreamResult:
    wav: np.ndarray
    block_times_ms: list[float] = field(default_factory=list)


def stream_convert(
    system: VCSystem,
    wav: np.ndarray,
    block_size: int,
    flush_ms: float = 600.0,
    warmup_blocks: int = 8,
) -> StreamResult:
    """Feed `wav` through `system` in fixed blocks, timing each one.

    Silence is appended so delayed output drains; the result keeps that tail so
    the scorer can estimate and remove the system's delay.
    """
    system.set_block_size(block_size)

    # Warm up (CUDA init, cudnn autotune) on real audio, then discard the state.
    system.reset()
    for i in range(warmup_blocks):
        chunk = wav[i * block_size : (i + 1) * block_size]
        if len(chunk) < block_size:
            break
        system.process_block(chunk.copy())
    system.sync()
    system.reset()

    flush = int(system.io_sr * flush_ms / 1000)
    padded = np.concatenate([wav, np.zeros(flush, dtype=np.float32)])
    n_blocks = int(np.ceil(len(padded) / block_size))
    padded = np.pad(padded, (0, n_blocks * block_size - len(padded)))

    out = np.zeros_like(padded)
    times: list[float] = []
    for i in range(n_blocks):
        block = padded[i * block_size : (i + 1) * block_size].copy()
        t0 = time.perf_counter()
        y = system.process_block(block)
        system.sync()
        times.append((time.perf_counter() - t0) * 1000.0)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        if len(y) != block_size:
            raise ValueError(
                f"{system.name}: process_block returned {len(y)} samples, expected {block_size}"
            )
        out[i * block_size : (i + 1) * block_size] = y
    return StreamResult(wav=out, block_times_ms=times)
