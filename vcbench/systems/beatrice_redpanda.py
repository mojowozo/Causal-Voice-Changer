"""Beatrice 2 baseline, driven through an existing Redpanda-vc install.

Run with that project's interpreter, e.g.

    PYTHONPATH=/mnt/ssd/VoiceChanger /mnt/ssd/Redpanda-vc/env/bin/python \
        -m vcbench.convert --system vcbench.systems.beatrice_redpanda:BeatriceRedpanda \
        --run beatrice --blocks 10,20,40

Training runs the stock Beatrice Trainer 2.0.0-rc.0 (through Redpanda-vc's
patched launcher) on all target voices as one multi-speaker model, which is how
the trainer is meant to be used. Inference uses Redpanda-vc's PyTorch port of
the streaming model, not the official closed-source runtime, so compute times
are not those of the official VST.

Nothing is written into the Redpanda-vc folder.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from vcbench.streaming import VCSystem, Voice


class BeatriceRedpanda(VCSystem):
    name = "beatrice"
    io_sr = 48000

    def __init__(
        self,
        root: str = "/mnt/ssd/Redpanda-vc",
        steps: int = 10000,
        train_gpu: str = "0",
        device: str = "cuda:0",
        vq_neighbors: int = 0,
        workers: int = 8,
        solo: list[str] | None = None,
    ):
        # solo: voice ids that get a single-speaker model of their own instead
        # of joining the shared multi-speaker one.
        self.solo = set(solo or [])
        self.root = Path(root).resolve()
        self.trainer = self.root / "models" / "beatrice" / "beatrice-trainer"
        self.steps, self.train_gpu, self.device = steps, train_gpu, device
        self.vq_neighbors, self.workers = vq_neighbors, workers
        self.semitones = 0
        self.rt = None
        self.model_dir: Path | None = None

    # ---- voice building -------------------------------------------------

    def _joint_dir(self, voice: Voice) -> Path:
        name = f"_solo_{voice.id}" if voice.id in self.solo else "_joint"
        return voice.workdir.resolve().parent / name

    def _export(self, joint: Path) -> Path | None:
        done = sorted((joint / "model").glob(f"paraphernalia_*_{self.steps:08d}"))
        return done[-1] if done else None

    def build_voices(self, voices: list[Voice]) -> None:
        groups: dict[Path, list[Voice]] = {}
        for voice in voices:
            groups.setdefault(self._joint_dir(voice), []).append(voice)
        for joint, members in groups.items():
            self._train(joint, members)

    def _train(self, joint: Path, voices: list[Voice]) -> None:
        if self._export(joint):
            return
        data = joint / "data" / "vcbench"
        for voice in voices:
            spk = data / voice.id
            spk.mkdir(parents=True, exist_ok=True)
            for wav in voice.train_wavs:
                link = spk / Path(wav).name
                if not link.exists():
                    link.symlink_to(Path(wav).resolve())
        out = joint / "model"
        out.mkdir(parents=True, exist_ok=True)
        config = json.loads((self.trainer / "assets" / "default_config.json").read_text())
        config |= {"n_steps": self.steps, "num_workers": self.workers}
        (joint / "config.json").write_text(json.dumps(config, indent=1))
        launcher = self.trainer / "beatrice_trainer" / "redpanda_main.py"
        if not launcher.exists():
            raise RuntimeError(f"{launcher} is missing; train one Beatrice model in Redpanda-vc first")
        cmd = [sys.executable, "-u", str(launcher), "-d", str(data), "-o", str(out), "-c", str(joint / "config.json")]
        if any(out.glob("checkpoint_latest.pt*")):
            cmd.append("-r")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=self.train_gpu, PYTHONPATH=str(self.root))
        with (joint / "build.log").open("a") as fh:
            proc = subprocess.run(cmd, cwd=self.root, env=env, stdout=fh, stderr=subprocess.STDOUT)
        if proc.returncode != 0 or not self._export(joint):
            raise RuntimeError(f"Beatrice training failed; see {joint / 'build.log'}")

    def build_voice(self, voice: Voice) -> None:
        if not self._export(self._joint_dir(voice)):
            raise RuntimeError("Beatrice voices are trained jointly; build_voices was not run")

    # ---- inference ------------------------------------------------------

    def load_voice(self, voice: Voice) -> None:
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))
        from beatrice.realtime import BeatriceRealtime

        export = self._export(self._joint_dir(voice))
        if self.rt is None or self.model_dir != export:
            self.rt = BeatriceRealtime(str(export), self.io_sr, device=self.device, vq_neighbors=self.vq_neighbors)
            self.model_dir = export
        names = [v["name"] for v in self.rt.model.voices]
        self.rt.model.set_speaker(names.index(voice.id))
        self.rt.reset()

    def set_pitch_shift(self, semitones: int) -> None:
        self.semitones = int(semitones)
        self.rt.model.set_pitch_shift(self.semitones)

    def reset(self) -> None:
        self.rt.reset()

    def process_block(self, block: np.ndarray) -> np.ndarray:
        return self.rt.process(block)

    def reported_latency_ms(self) -> float:
        return self.rt.latency_seconds * 1000.0

    def sync(self) -> None:
        if self.device.startswith("cuda"):
            import torch

            torch.cuda.synchronize()
