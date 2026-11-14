"""RVC baseline, driven through an existing Applio install.

Run with Applio's own interpreter, e.g.

    PYTHONPATH=/mnt/ssd/VoiceChanger /mnt/ssd/Applio-3.7.1/.venv/bin/python \
        -m vcbench.convert --system vcbench.systems.rvc_applio:RvcApplio --run rvc ...

The Applio install is only read: training artifacts go to the voice workdir via
APPLIO_LOGS_DIR, never into Applio's own logs folder.

Streaming uses Applio's realtime VoiceChanger (chunked inference with extra
context, SOLA alignment and crossfade), block by block, exactly as its realtime
tab does. Offline uses its normal whole-file inference.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from vcbench import audio
from vcbench.streaming import VCSystem, Voice

IO_SR = 48000


class RvcApplio(VCSystem):
    name = "rvc"
    io_sr = IO_SR

    def __init__(
        self,
        applio_root: str = "/mnt/ssd/Applio-3.7.1",
        sample_rate: int = 48000,
        vocoder: str = "HiFi-GAN",
        epochs: int = 200,
        batch_size: int = 8,
        gpu: str = "0",
        f0_method: str = "rmvpe",
        embedder: str = "contentvec",
        crossfade_s: float = 0.05,
        extra_s: float = 2.5,
        index_rate: float = 0.5,
        protect: float = 0.5,
        prebuilt: dict[str, dict[str, str]] | None = None,
    ):
        # prebuilt: {voice id: {"model": weights.pth, "index": file.index}} for
        # voices that already have an RVC build and should not be retrained.
        self.prebuilt = prebuilt or {}
        self.root = Path(applio_root).resolve()
        self.sample_rate, self.vocoder, self.epochs = sample_rate, vocoder, epochs
        self.batch_size, self.gpu = batch_size, gpu
        self.f0_method, self.embedder = f0_method, embedder
        self.crossfade_s, self.extra_s = crossfade_s, extra_s
        self.index_rate, self.protect = index_rate, protect
        self.semitones = 0
        self.vc = None
        self.model_path = self.index_path = None
        self._offline = None

    # ---- voice building -------------------------------------------------

    def _artifacts(self, voice: Voice) -> tuple[Path | None, Path | None]:
        if voice.id in self.prebuilt:
            pre = self.prebuilt[voice.id]
            return Path(pre["model"]), (Path(pre["index"]) if pre.get("index") else None)
        model_dir = voice.workdir.resolve() / voice.id
        weights = sorted(model_dir.glob(f"{voice.id}_*e_*s.pth"), key=lambda p: p.stat().st_mtime)
        index = model_dir / f"{voice.id}.index"
        return (weights[-1] if weights else None), (index if index.exists() else None)

    def _cli(self, voice: Voice, *args: str) -> None:
        env = dict(os.environ, APPLIO_LOGS_DIR=str(voice.workdir.resolve()))
        log = voice.workdir / "build.log"
        with log.open("a") as fh:
            fh.write(f"\n$ core.py {' '.join(args)}\n")
            fh.flush()
            proc = subprocess.run(
                [sys.executable, "core.py", *args], cwd=self.root, env=env, stdout=fh, stderr=subprocess.STDOUT
            )
        if proc.returncode != 0:
            raise RuntimeError(f"Applio `{args[0]}` failed for {voice.id}; see {log}")

    def build_voice(self, voice: Voice) -> None:
        weights, index = self._artifacts(voice)
        if (weights and index) or voice.id in self.prebuilt:
            return
        dataset = voice.workdir.resolve() / "dataset"
        dataset.mkdir(parents=True, exist_ok=True)
        for wav in voice.train_wavs:
            link = dataset / Path(wav).name
            if not link.exists():
                link.symlink_to(Path(wav).resolve())
        sr = str(self.sample_rate)
        self._cli(voice, "preprocess", "--model-name", voice.id, "--dataset-path", str(dataset), "--sample-rate", sr)
        self._cli(
            voice, "extract", "--model-name", voice.id, "--sample-rate", sr,
            "--f0-method", self.f0_method, "--embedder-model", self.embedder, "--gpu", self.gpu,
        )
        self._cli(
            voice, "train", "--model-name", voice.id, "--sample-rate", sr, "--vocoder", self.vocoder,
            "--total-epoch", str(self.epochs), "--save-every-epoch", str(min(100, self.epochs)),
            "--save-only-latest", "--batch-size", str(self.batch_size), "--gpu", self.gpu,
        )
        weights, index = self._artifacts(voice)
        if weights and not index:
            self._cli(voice, "index", "--model-name", voice.id)
            weights, index = self._artifacts(voice)
        if not (weights and index):
            raise RuntimeError(f"Applio finished but no weights/index for {voice.id} in {voice.workdir}")

    # ---- inference ------------------------------------------------------

    def _enter_applio(self) -> None:
        # Applio resolves its models and imports relative to the working directory.
        if Path.cwd() != self.root:
            os.chdir(self.root)
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))

    def load_voice(self, voice: Voice) -> None:
        weights, index = self._artifacts(voice)
        if not weights:
            raise RuntimeError(f"voice {voice.id} is not built")
        self.model_path, self.index_path = str(weights), str(index or "")
        self.vc = None  # rebuilt lazily for the next block size

    def set_pitch_shift(self, semitones: int) -> None:
        self.semitones = int(semitones)

    def set_block_size(self, n_samples: int) -> None:
        self._enter_applio()
        if self.vc is None:
            from rvc.realtime.core import VoiceChanger

            self.vc = VoiceChanger(
                block_frame=n_samples,
                cross_fade_overlap_size=self.crossfade_s,
                extra_convert_size=self.extra_s,
                model_path=self.model_path,
                index_path=self.index_path,
                f0_method=self.f0_method,
                embedder_model=self.embedder,
                silent_threshold=-90,
            )
        elif self.vc.block_frame != n_samples:
            vc = self.vc
            vc.block_frame = n_samples
            vc.vc_model.realloc(n_samples, vc.extra_frame, vc.crossfade_frame, vc.sola_search_frame)
            vc.generate_strength()

    def reset(self) -> None:
        vc, model = self.vc, self.vc.vc_model
        for buf in (model.audio_buffer, model.convert_buffer, model.pitch_buffer, model.pitchf_buffer, vc.sola_buffer):
            buf.zero_()
        vc._tail_remaining = 0

    def process_block(self, block: np.ndarray) -> np.ndarray:
        out, _vol, _perf = self.vc.on_request(
            block, f0_up_key=self.semitones, index_rate=self.index_rate, protect=self.protect
        )
        return np.asarray(out, dtype=np.float32)

    def convert_offline(self, wav: np.ndarray) -> np.ndarray:
        self._enter_applio()
        if self._offline is None:
            from rvc.infer.infer import VoiceConverter

            self._offline = VoiceConverter()
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = os.path.join(tmp, "in.wav"), os.path.join(tmp, "out.wav")
            audio.save(src, wav, IO_SR)
            self._offline.convert_audio(
                audio_input_path=src, audio_output_path=dst,
                model_path=self.model_path, index_path=self.index_path,
                pitch=self.semitones, f0_method=self.f0_method,
                index_rate=self.index_rate, protect=self.protect,
                embedder_model=self.embedder, export_format="WAV",
            )
            out, _ = audio.load(dst, IO_SR)
        return out

    def sync(self) -> None:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
