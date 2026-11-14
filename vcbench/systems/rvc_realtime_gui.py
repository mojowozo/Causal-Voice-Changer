"""RVC baseline through the user's own realtime app (RVC-Realtime-GUI-linux).

Run with the app's interpreter:

    PYTHONPATH=/mnt/ssd/VoiceChanger /mnt/ssd/RVC-Realtime-GUI-linux/.venv/bin/python \
        -m vcbench.convert --system vcbench.systems.rvc_realtime_gui:RvcRealtimeGui \
        --run rvc-app --blocks 40,80,100

The engine is driven headlessly, one audio callback per block, the way the
app's own tools/*_bench.py scripts do: no audio device is opened. It reads the
app's global settings (configs/engine.json) but nothing is written back: the
voice-pitch histogram is not updated and settings are never saved.

Voices are the Applio builds of the `rvc` run (link runs/<run>/voices to
../rvc/voices) or `prebuilt` ones; this adapter does not train.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from vcbench.streaming import Voice
from vcbench.systems.rvc_applio import RvcApplio


class RvcRealtimeGui(RvcApplio):
    name = "rvc-app"
    io_sr = 48000

    def __init__(
        self,
        app_root: str = "/mnt/ssd/RVC-Realtime-GUI-linux",
        gpu: str = "0",
        extra_s: float | None = None,  # None: the app's saved setting
        index_rate: float = 0.0,  # the app's default
        protect: float = 0.33,
        cuda_graph: bool = False,
        prebuilt: dict[str, dict[str, str]] | None = None,
    ):
        super().__init__(prebuilt=prebuilt)
        self.app = Path(app_root).resolve()
        self.gpu, self.app_extra_s = gpu, extra_s
        self.index_rate, self.protect = index_rate, protect
        self.cuda_graph = cuda_graph
        self.engine = None
        self.model = None
        self.block_ms = None

    def build_voice(self, voice: Voice) -> None:
        if not self._artifacts(voice)[0]:
            raise RuntimeError(f"voice {voice.id} has no RVC build; build it with the rvc run first")

    def _engine(self):
        if self.engine is None:
            os.environ["RVC_CUDA_GRAPH"] = "1" if self.cuda_graph else "0"
            os.chdir(self.app)
            sys.path.insert(0, str(self.app))
            from engine.core import RealtimeEngine
            from engine.settings import MODEL_SETTING_KEYS, EngineSettings

            eng = RealtimeEngine(lambda event, data: None)
            defaults = EngineSettings()
            for key in MODEL_SETTING_KEYS:  # not the user's per-model tuning
                if hasattr(defaults, key):
                    setattr(eng.settings, key, getattr(defaults, key))
            s = eng.settings
            s.gpu, s.index_rate, s.protect = self.gpu, self.index_rate, self.protect
            if self.app_extra_s is not None:
                s.extra_time = self.app_extra_s
            # Held (silent) blocks skip inference; keep every block comparable.
            s.hold_context = False
            s.input_denoise = s.output_denoise = False
            eng.voice_pitch.add = lambda *a, **k: None
            eng.function = "vc"
            eng.require_assets(s.f0method)
            eng.apply_selected_gpu()
            self.engine = eng
        return self.engine

    def load_voice(self, voice: Voice) -> None:
        weights, index = self._artifacts(voice)
        if not weights:
            raise RuntimeError(f"voice {voice.id} is not built")
        use_index = index if self.index_rate > 0 else None
        self.model = SimpleNamespace(name=voice.id, model_path=Path(weights), index_path=use_index)
        self._engine().load_model(self.model)
        self.block_ms = None

    def set_pitch_shift(self, semitones: int) -> None:
        self.semitones = int(semitones)
        self.engine.rvc.change_key(self.semitones)

    def set_block_size(self, n_samples: int) -> None:
        block_ms = round(1000 * n_samples / self.io_sr)
        if block_ms % 10:
            raise ValueError("the app's block length is a multiple of 10 ms")
        self.block_ms = block_ms
        self.engine.settings.block_time = block_ms / 1000
        self._prepare()

    def _prepare(self) -> None:
        eng = self.engine
        for cache in (eng.rvc.cache_pitch, eng.rvc.cache_pitchf):
            cache.zero_()
        eng.prepare_inference(self.io_sr)
        eng.rvc.change_key(self.semitones)

    def reset(self) -> None:
        self._prepare()

    def process_block(self, block: np.ndarray) -> np.ndarray:
        out = self.engine.audio_callback(block[:, None].copy(), len(block), None, None)
        return np.asarray(out, dtype=np.float32)

    def convert_offline(self, wav: np.ndarray):
        return None

    def reported_latency_ms(self):
        return None
