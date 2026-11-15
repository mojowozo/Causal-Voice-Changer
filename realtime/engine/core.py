"""Headless realtime engine for CausalVC voices.

Model and device discovery, settings, stream routing, conversion, file input
and recording.  Front ends talk to it through ``engine.server``.

A CausalVC voice converts each block as it arrives and its output is already
continuous from block to block, so there is no context window, splice or
pitch cache here: a block goes in, the same number of samples comes out.

Threading: every public method is called from one command thread (the
server serializes commands and ``tick``).  PortAudio callbacks, the native
JACK worker, and the file decoder only touch the audio buffers and the
``latest_*`` meter values.
"""

import os
import queue
import shutil
import subprocess
import threading
import time
import traceback

import librosa
import numpy as np
import sounddevice as sd
import torch

from engine.devices import DeviceCatalog
from engine.gpu import Device, backend_label, cuda_indices, default_index, gpu_name
from engine.settings import (
    DEFAULT_RECORDING_FOLDER,
    MODEL_SETTING_KEYS,
    PERFORMANCE_DEFAULT_KEYS,
    PROJECT_ROOT,
    RESETTABLE_MODEL_SETTING_KEYS,
    RESTART_SETTING_KEYS,
    EngineSettings,
    coerce_setting,
    load_model_settings,
    load_settings,
    save_model_settings,
    save_settings,
)
from engine.version import APP_TITLE, BUILD_LABEL
from infer import causalvc_backend
from tools.audio_fifo import AudioFrameFifo, enqueue_latest
from tools.audio_routing import (
    ASIO_API,
    check_options,
    is_native_api,
    scatter_mono,
    select_channels,
    stream_options,
)
from tools.file_audio_source import FileAudioSource
from tools.model_import import ModelImportError, import_models, install_loose, voice_files
from tools import asio as asio_driver
from tools import hushwire, rnnoise
from tools.limiter import soft_clip
from tools.loudness import TARGETS_LUFS, LoudnessMeter
from tools.model_registry import (
    TrashError,
    discover_models,
    model_name_problem,
    move_to_trash,
    rename_model_folder,
)
from tools.wav_recorder import WavRecorder

MODELS_ROOT = os.path.join(PROJECT_ROOT, "models")
MODEL_SETTINGS_SAVE_DELAY_SECONDS = 0.25
METER_INTERVAL_SECONDS = 0.1
#: RNNoise's own output delay: one 10 ms frame of buffering plus its 10 ms
#: window overlap (960 samples, measured on real speech; tests/test_rnnoise.py).
RNNOISE_DELAY_SECONDS = 0.020
#: Audio queued for an output beyond one chunk plus this margin is a standing
#: backlog: a moment when inference fell behind leaves extra audio queued
#: that, with input and output running at the same rate, never drains.  It
#: is trimmed by dropping blocks of pure silence (see ``queue_output``).  The
#: margin is at least twice the recent inference time, so trimming never
#: leaves the output at risk of running dry.
BACKLOG_MARGIN_SECONDS = 0.060
#: How often a growing underrun count is logged.
UNDERRUN_LOG_SECONDS = 5.0
#: A device latency above this is a bogus report, not a real buffer: some
#: PipeWire JACK nodes make PortAudio report 2**32 - period frames (~89478 s).
MAX_PLAUSIBLE_STREAM_LATENCY = 2.0
FUNCTIONS = {"vc": "vc", "passthrough": "im"}


class EngineError(Exception):
    """An error a front end can show; ``code`` is stable and translatable."""

    def __init__(self, code, message, **details):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


def printt(strr, *args):
    if len(args) == 0:
        print(strr)
    else:
        print(strr % args)


def plausible_stream_latency(latency):
    """The device's reported latency in seconds, or 0.0 if it is impossible."""
    if isinstance(latency, (tuple, list)):
        latency = latency[-1]
    try:
        latency = float(latency)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(latency) or not 0.0 <= latency <= MAX_PLAUSIBLE_STREAM_LATENCY:
        printt("Ignoring the device's reported latency of %.3f s: not plausible", latency)
        return 0.0
    return latency


def peak_meter(samples):
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    peak_db = 20.0 * np.log10(max(peak, 1e-6))
    return float(np.clip((peak_db + 60.0) / 60.0, 0.0, 1.0))


class RealtimeEngine:
    def __init__(self, emit):
        """``emit(event, payload)`` delivers events to connected clients."""
        self.emit = emit
        self.config = Device()
        self.auto_gpu_index = default_index()
        printt(APP_TITLE)
        printt(BUILD_LABEL)
        self.running = False
        self.function = "vc"
        self.delay_time = 0.0
        self.samplerate = None
        self.input_channels = 1
        self.output_channels = 2
        self.input_api = ""
        self.output_api = ""
        self.monitor_api = ""
        self.input_selectors = []
        self.output_selectors = []
        self.monitor_selectors = []
        self.monitor_device_index = None
        #: The monitor is on the same ASIO device as the input or output and shares its stream.
        self.monitor_shares_asio = False
        #: What the ASIO driver accepts, per PortAudio device (read while idle).
        self.asio_sizes = {}
        #: The ASIO buffer the driver actually delivers, in samples (from its callbacks).
        self.asio_frames = None
        self.reported_asio_frames = None
        #: The latency asked of ASIO streams this run ("low" or seconds).
        self.asio_latency = None
        self.stream = None
        self.input_stream = None
        self.output_stream = None
        self.monitor_stream = None
        self.output_queue = None
        self.monitor_queue = None
        self.native_input_fifo = None
        self.native_worker_stop = None
        self.native_worker_wakeup = None
        self.native_worker = None
        self.last_input_meter_update = 0.0
        self.last_output_meter_update = 0.0
        self.latest_input_meter = 0.0
        self.latest_output_meter = 0.0
        self.latest_monitor_meter = 0.0
        self.latest_infer_time = 0
        #: Silence dropped to drain a standing output backlog, in frames.
        self.trimmed_frames = 0
        #: Output callbacks that found less audio queued than they needed,
        #: counted once audio has started flowing (reset per stream).
        self.output_underruns = 0
        self.logged_underruns = 0
        self.underrun_logged_at = 0.0
        self.output_flowing = False
        #: The input denoiser in use (RNNoise or Hushwire), or None; read by the audio callback.
        self.active_denoiser = None
        self.active_denoise_delay = 0.0
        #: Denoisers built for the current stream, by method (switching back keeps their state).
        self.denoisers = {}
        self.denoise_lock = threading.RLock()
        #: Hushwire's model download: None, "downloading" or "failed".
        self.hushwire_download = None
        self.recorder = WavRecorder()
        self.file_audio_source = None
        self.file_input_path = ""
        self.file_state = (False, False)
        self.pending_model_settings_name = None
        self.model_settings_save_due = 0.0
        self.ffmpeg_path = shutil.which("ffmpeg") or os.path.join(
            PROJECT_ROOT, "tools", "ffmpeg", "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
        )
        os.makedirs(MODELS_ROOT, exist_ok=True)
        self.devices = DeviceCatalog(sd)
        self.refresh_gpu_options()
        self.refresh_models()
        self.devices.refresh()
        self.settings = load_settings()
        self.normalize_settings()
        self.settings.__dict__.update(self.model_settings_for(self.settings.model_name))
        self.log_startup_diagnostics()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    def state(self):
        source = self.file_audio_source
        return {
            "running": self.running,
            "function": "passthrough" if self.function == "im" else "vc",
            "settings": self.settings.to_dict(),
            "models": [
                {
                    "name": model.name,
                    "model_file": model.model_path.name,
                    "size_bytes": model.size_bytes,
                    "modified": int(model.modified),
                }
                for model in self.models
            ],
            "devices": self.devices.to_dict(),
            "gpus": self.gpu_options,
            "samplerate": self.samplerate if self.running else None,
            "delay_ms": int(np.round(self.delay_time * 1000)) if self.running else None,
            "recording": self.recorder.active,
            "file": {
                "path": self.file_input_path or None,
                "duration": source.duration if source else None,
                "playing": bool(source and source.playing),
            },
            "ffmpeg": os.path.isfile(self.ffmpeg_path),
            "rnnoise": rnnoise.available(),
            "hushwire": self.hushwire_status(),
            "asio": self.asio_state(),
        }

    def emit_state(self):
        self.emit("state", self.state())

    def status(self, code, **details):
        self.emit("status", {"code": code, **details})

    # ------------------------------------------------------------------
    # Models, GPUs, devices
    # ------------------------------------------------------------------
    def refresh_models(self):
        installed, problems = install_loose(MODELS_ROOT)
        if installed:
            printt("Installed voice file(s) from the models folder: %s", ", ".join(installed))
        for problem in problems:
            printt("Skipped a voice file in the models folder: %s", problem)
        self.models = discover_models(MODELS_ROOT)
        self.models_by_name = {model.name: model for model in self.models}

    def model_settings_for(self, model_name):
        model = self.models_by_name.get(model_name)
        return load_model_settings(str(model.directory) if model else None)

    def refresh_gpu_options(self):
        self.gpu_options = [{"id": "auto", "label": None}]
        for index in cuda_indices():
            self.gpu_options.append(
                {
                    "id": str(index),
                    "label": f"GPU {index} — {gpu_name(index)}",
                }
            )

    def apply_selected_gpu(self):
        gpu = self.settings.gpu
        index = self.auto_gpu_index if gpu == "auto" else int(gpu)
        if index is None:
            return
        try:
            self.config.select_cuda_device(index)
        except ValueError as error:
            raise EngineError("gpu_invalid", str(error)) from error

    def normalize_settings(self):
        """Point saved model/device/GPU choices at ones that exist now."""
        settings = self.settings
        if settings.model_name not in self.models_by_name:
            settings.model_name = self.models[0].name if self.models else ""
        settings.input_device = self.devices.resolve(settings.input_device, "input")
        settings.output_device = self.devices.resolve(settings.output_device, "output")
        settings.monitor_device = self.devices.resolve_monitor(settings.monitor_device)
        if settings.gpu not in {option["id"] for option in self.gpu_options}:
            settings.gpu = "auto"

    def reload_models(self):
        self.flush_model_settings_save(force=True)
        previous = self.settings.model_name
        self.refresh_models()
        self.normalize_settings()
        if self.settings.model_name != previous and self.running:
            self.stop()
            self.status("settings_changed")
        self.settings.__dict__.update(self.model_settings_for(self.settings.model_name))
        self.emit_state()

    def import_model(self, paths):
        """Install dropped voice files (.cvc, or older .causalvc.json) into models/ and select the first."""
        if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
            raise EngineError("bad_request", "paths must be a list of strings")
        self.status("importing_model")
        try:
            names = import_models(paths, MODELS_ROOT)
        except ModelImportError as error:
            raise EngineError(error.code, error.message) from error
        printt("Imported model(s): %s", ", ".join(names))
        self.refresh_models()
        self.update_settings({"model_name": names[0]})
        self.status("model_imported", names=names)
        self.emit_state()
        return names

    def export_model(self, name, path):
        """Pack a model folder into one .cvc file at ``path``, with its saved settings, to share."""
        model = self.models_by_name.get(name)
        if model is None:
            raise EngineError("no_model", f"No model named {name}.")
        if not isinstance(path, str) or not os.path.isabs(path):
            raise EngineError("bad_request", "path must be an absolute file path")
        if not path.lower().endswith(voice_files.CVC_SUFFIX):
            path += voice_files.CVC_SUFFIX
        self.flush_model_settings_save(force=True)  # the settings file is packed too
        try:
            written = voice_files.folder_to_cvc(model.directory, path, settings=True)
        except (OSError, ValueError, KeyError) as error:
            raise EngineError("export_failed", f"Could not write {path}: {error}") from error
        printt("Exported %s to %s", name, written)
        self.status("model_exported", name=name, path=str(written))
        return str(written)

    def delete_model(self, name):
        """Move a model's folder to the Trash; another model is selected if needed."""
        model = self.models_by_name.get(name)
        if model is None:
            raise EngineError("no_model", f"Unknown model: {name}")
        if self.running and self.function == "vc" and name == self.settings.model_name:
            raise EngineError("model_in_use", "Stop converting before deleting this model.")
        # A pending settings save would recreate the folder after it is gone.
        self.flush_model_settings_save(force=True)
        try:
            move_to_trash(model.directory)
        except (OSError, subprocess.SubprocessError, TrashError) as error:
            raise EngineError("delete_failed", str(error), model=name) from error
        printt("Moved model to the Trash: %s", model.directory)
        self.reload_models()
        save_settings(self.settings)
        self.status("model_deleted", name=name)

    def rename_model(self, name, new_name):
        """Rename a model's folder (its name everywhere); settings go with it."""
        model = self.models_by_name.get(name)
        if model is None:
            raise EngineError("no_model", f"Unknown model: {name}")
        if new_name == name:
            return
        problem = model_name_problem(new_name)
        if problem is not None:
            raise EngineError("bad_model_name", f"Invalid model name: {new_name!r}", reason=problem)
        clash = next(
            (other for other in self.models_by_name
             if other != name and other.casefold() == new_name.casefold()),
            None,
        )
        if clash is not None:
            raise EngineError("model_name_taken", f"A model named {clash} already exists.")
        if self.running and self.function == "vc" and name == self.settings.model_name:
            raise EngineError("model_in_use", "Stop converting before renaming this model.")
        # A pending settings save would recreate the old folder.
        self.flush_model_settings_save(force=True)
        try:
            rename_model_folder(model.directory, new_name)
        except FileExistsError as error:
            raise EngineError("model_name_taken", f"{new_name} already exists.") from error
        except OSError as error:
            raise EngineError("rename_failed", str(error), model=name) from error
        printt("Renamed model %s to %s", name, new_name)
        if self.settings.model_name == name:
            self.settings.model_name = new_name
        self.reload_models()
        save_settings(self.settings)
        self.status("model_renamed", name=name, new_name=new_name)

    def reload_devices(self):
        if self.running:
            self.stop()
            self.status("settings_changed")
        self.devices.refresh()
        self.asio_sizes.clear()
        self.normalize_settings()
        save_settings(self.settings)
        self.emit_state()

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------
    def update_settings(self, changes):
        if not isinstance(changes, dict) or not changes:
            raise EngineError("bad_request", "settings must be a non-empty object")
        coerced = {}
        for key, value in changes.items():
            try:
                coerced[key] = coerce_setting(key, value)
            except ValueError as error:
                raise EngineError("bad_setting", str(error), key=key) from error
        if "model_name" in coerced and coerced["model_name"] not in self.models_by_name:
            raise EngineError("no_model", f"Unknown model: {coerced['model_name']}")
        for key in ("input_device", "output_device"):
            endpoints = self.devices.inputs if key == "input_device" else self.devices.outputs
            if key in coerced and coerced[key] not in endpoints:
                raise EngineError("bad_device", f"Unknown device: {coerced[key]}", key=key)
        if coerced.get("monitor_device") and coerced["monitor_device"] not in self.devices.outputs:
            raise EngineError("bad_device", "Unknown monitor device", key="monitor_device")
        if "gpu" in coerced and coerced["gpu"] not in {o["id"] for o in self.gpu_options}:
            raise EngineError("gpu_invalid", f"Unknown GPU: {coerced['gpu']}")

        changed = {
            key: value
            for key, value in coerced.items()
            if getattr(self.settings, key) != value
        }
        if not changed:
            self.emit_state()
            return
        if "model_name" in changed:
            self.flush_model_settings_save(force=True)
        for key, value in changed.items():
            setattr(self.settings, key, value)
        if "model_name" in changed:
            # Per-model settings follow the newly selected model.
            self.settings.__dict__.update(self.model_settings_for(changed["model_name"]))
        elif any(key in MODEL_SETTING_KEYS for key in changed):
            self.schedule_model_settings_save()
        self.apply_hot_settings(changed)
        if self.running and any(key in RESTART_SETTING_KEYS for key in changed):
            self.stop()
            self.status("settings_changed")
        if any(key not in MODEL_SETTING_KEYS for key in changed):
            save_settings(self.settings)
        self.emit_state()

    def apply_hot_settings(self, changed):
        if not hasattr(self, "voice"):
            return
        if "pitch" in changed:
            self.voice.change_key(self.settings.pitch)
        if "input_gain_db" in changed and getattr(self, "loudness", None) is not None:
            # The model follows the input level: earlier readings no longer apply.
            self.loudness.reset()
        if {"input_denoise", "input_denoise_method"} & set(changed):
            self.refresh_denoiser()

    def reset_settings(self, group):
        defaults = EngineSettings()
        if group == "general":
            keys = RESETTABLE_MODEL_SETTING_KEYS
        elif group == "performance":
            keys = PERFORMANCE_DEFAULT_KEYS
        else:
            raise EngineError("bad_request", "group must be general or performance")
        self.update_settings({key: getattr(defaults, key) for key in keys})
        if group == "general":
            self.flush_model_settings_save(force=True)

    def schedule_model_settings_save(self):
        if self.settings.model_name not in self.models_by_name:
            return
        self.pending_model_settings_name = self.settings.model_name
        self.model_settings_save_due = time.monotonic() + MODEL_SETTINGS_SAVE_DELAY_SECONDS

    def flush_model_settings_save(self, force=False):
        model_name = self.pending_model_settings_name
        if not model_name:
            return
        if not force and time.monotonic() < self.model_settings_save_due:
            return
        model = self.models_by_name.get(model_name)
        if model is not None:
            try:
                save_model_settings(str(model.directory), self.settings)
            except OSError as error:
                printt("Could not save model settings: %s", error)
        self.pending_model_settings_name = None
        self.model_settings_save_due = 0.0

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def log_startup_diagnostics(self):
        """Log release-relevant checks without opening any audio stream."""
        printt("=== Startup diagnostics ===")
        printt("FFmpeg: %s (found=%s)", self.ffmpeg_path, os.path.isfile(self.ffmpeg_path))
        printt("Models discovered: %s", len(self.models))
        printt("GPU backend: %s (available=%s)", backend_label(), torch.cuda.is_available())
        if torch.cuda.is_available():
            printt("GPUs: %s", ", ".join(gpu_name(index) for index in cuda_indices()))
            printt("Selected GPU: %s", self.settings.gpu)
        try:
            os.makedirs(self.settings.recording_folder, exist_ok=True)
            printt(
                "Recording folder writable: %s",
                os.access(self.settings.recording_folder, os.W_OK),
            )
        except OSError as error:
            printt("Recording folder unavailable: %s", error)
        printt(
            "Configured route: input=%s / output=%s / monitor=%s",
            self.settings.input_device,
            self.settings.output_device,
            self.settings.monitor_device or "Disabled",
        )
        printt("=== Startup diagnostics complete ===")

    def log_active_audio_route(self):
        printt("=== Active audio route ===")
        printt(
            "Input=%s / Output=%s / Monitor=%s",
            self.input_api or "File",
            self.output_api,
            self.monitor_api or "Disabled",
        )
        printt(
            "Rate=%s Hz / Chunk=%.3f sec / PIPEWIRE_QUANTUM=%s",
            self.samplerate,
            self.settings.block_time,
            os.environ.get("PIPEWIRE_QUANTUM", "(server default)"),
        )
        printt(
            "Channel selectors: input=%s output=%s monitor=%s",
            self.input_selectors,
            self.output_selectors,
            self.monitor_selectors,
        )
        printt("=== Active audio route complete ===")

    # ------------------------------------------------------------------
    # Periodic work
    # ------------------------------------------------------------------
    def tick(self):
        """Called about 20 times per second from the command thread."""
        self.flush_model_settings_save()
        self.log_underruns()
        if self.asio_frames != self.reported_asio_frames:
            self.reported_asio_frames = self.asio_frames
            if self.asio_frames is not None:
                printt("ASIO buffer in use: %s samples", self.asio_frames)
            self.emit_state()
        if self.running and not self.streams_active():
            printt("Audio stream stopped unexpectedly; returning to the idle state.")
            self.stop()
            self.status("stream_stopped")
        source = self.file_audio_source
        if source is not None:
            if source.error:
                error, source.error = source.error, None
                self.emit("error", {"code": "file_error", "message": str(error)})
            file_state = (bool(source.playing and not source.finished), source.finished)
            if file_state != self.file_state:
                self.file_state = file_state
                self.emit_state()
        if self.running or (source is not None and source.playing):
            self.emit("meters", self.meters())

    def streams_active(self):
        def active(stream):
            return stream is not None and bool(getattr(stream, "active", False))

        if self.file_source_active:
            # File input owns no PortAudio input stream: decoded blocks are
            # fed into audio_callback by FileAudioSource.
            return active(self.output_stream)
        return active(self.input_stream) and active(self.output_stream)

    def meters(self):
        recording_seconds = None
        if self.recorder.active and self.recorder.started_at is not None:
            recording_seconds = (
                time.time() - self.recorder.started_at.timestamp()
            )
        source = self.file_audio_source
        return {
            "input": self.latest_input_meter if self.running else 0.0,
            "output": self.latest_output_meter if self.running else 0.0,
            "monitor": self.latest_monitor_meter if self.running else 0.0,
            "infer_ms": self.latest_infer_time if self.running else None,
            "queued_ms": self.queued_ms(),
            "underruns": self.output_underruns if self.running else None,
            "recording_seconds": recording_seconds,
            "file_position": source.position if source else None,
            "loudness": self.output_loudness() if self.running else None,
        }

    def output_loudness(self):
        """The output-level reading at the current gain (tools/loudness.py)."""
        meter = getattr(self, "loudness", None)
        if meter is None:
            return None
        gain_db = self.settings.output_gain_db
        if self.file_source_active:
            volume = float(np.clip(self.settings.file_input_volume, 0.0, 1.0))
            gain_db += 20 * np.log10(max(volume, 1e-6))
        return meter.summary(gain_db, TARGETS_LUFS[self.settings.loudness_target])

    def shutdown(self):
        self.flush_model_settings_save(force=True)
        self.stop()

    # ------------------------------------------------------------------
    # Start / stop
    # ------------------------------------------------------------------
    @property
    def file_source_active(self):
        return self.settings.input_source == "file"

    def start(self, function="vc"):
        if function not in FUNCTIONS:
            raise EngineError("bad_request", "function must be vc or passthrough")
        if self.running:
            self.stop()
        self.function = FUNCTIONS[function]
        model = self.validate_start()
        self.status("preparing")
        try:
            self.start_vc(model)
        except EngineError:
            self.stop()
            raise
        except Exception as error:
            printt(traceback.format_exc())
            self.stop()
            raise EngineError("audio_start_failed", str(error)) from error
        if self.stream is not None:
            self.delay_time = (
                plausible_stream_latency(self.stream.latency)
                + self.settings.block_time
                + (self.voice.delay_seconds if self.function == "vc" else 0.0)
            )
        self.refresh_denoiser()  # prepare_inference cleared the last stream's denoiser
        save_settings(self.settings)
        self.status(
            "passthrough_started" if self.function == "im" else "conversion_started"
        )
        self.emit_state()

    def validate_start(self):
        model = self.models_by_name.get(self.settings.model_name)
        if model is None:
            raise EngineError("no_model", "Select a voice from the models folder.")
        if not model.model_path.is_file():
            raise EngineError(
                "model_file_missing",
                f"Voice file not found: {model.model_path}",
                path=str(model.model_path),
            )
        self.apply_selected_gpu()
        if self.file_source_active:
            if not self.file_input_path:
                raise EngineError("no_audio_file", "Select an audio file first.")
            if not os.path.isfile(self.ffmpeg_path):
                raise EngineError("ffmpeg_missing", "FFmpeg was not found.")
        self.set_devices()
        return model

    def set_devices(self):
        settings = self.settings
        output = self.devices.outputs.get(settings.output_device)
        if output is None:
            raise EngineError("bad_device", "Select an output device.", key="output_device")
        if self.file_source_active:
            # A decoded file has no PortAudio input endpoint.  Binding the
            # unused input side to the output endpoint prevents a stale or
            # disabled microphone from vetoing conversion startup.
            input_index = output.index
            self.input_api = "File"
            self.input_selectors = []
        else:
            endpoint = self.devices.inputs.get(settings.input_device)
            if endpoint is None:
                raise EngineError("bad_device", "Select an input device.", key="input_device")
            input_index = endpoint.index
            self.input_api = endpoint.api
            self.input_selectors = endpoint.selectors
        sd.default.device = (input_index, output.index)
        self.output_api = output.api
        self.output_selectors = output.selectors
        monitor = self.devices.outputs.get(settings.monitor_device or "")
        self.monitor_device_index = monitor.index if monitor else None
        self.monitor_selectors = monitor.selectors if monitor else []
        if (
            self.monitor_device_index == output.index
            and self.monitor_selectors == self.output_selectors
        ):
            printt(
                "Monitor output is the same as the main output; "
                "the duplicate stream was omitted."
            )
            self.monitor_device_index = None
            self.monitor_selectors = []
        self.monitor_api = monitor.api if self.monitor_device_index is not None else ""
        self.check_asio_route(input_index, output.index)
        printt(
            "Audio route: Input [%s] %s / Output [%s] %s / Monitor [%s] %s",
            self.input_api,
            settings.input_device,
            self.output_api,
            settings.output_device,
            self.monitor_api or "Disabled",
            settings.monitor_device or "Disabled",
        )

    def check_asio_route(self, input_index, output_index):
        """PortAudio's ASIO host opens one stream on one driver at a time.

        Input, output and monitor on the same ASIO device therefore share one
        stream (``open_asio_stream``); a second ASIO device cannot be used
        alongside the first.
        """
        asio_index = None
        for api, index in ((self.input_api, input_index), (self.output_api, output_index)):
            if api != ASIO_API:
                continue
            if asio_index is not None and index != asio_index:
                raise EngineError(
                    "asio_one_device",
                    "ASIO can only use one device at a time. Choose the same ASIO device for "
                    "input and output, or a [WASAPI] device for one of them.",
                    key="output_device",
                )
            asio_index = index
        self.monitor_shares_asio = False
        if self.monitor_api != ASIO_API or asio_index is None:
            return
        if self.monitor_device_index == asio_index:
            self.monitor_shares_asio = True
            return
        printt("The monitor is on a second ASIO device; it was turned off for this run.")
        self.emit(
            "error",
            {
                "code": "monitor_failed",
                "message": "ASIO can only use one device at a time; "
                "choose a [WASAPI] device or the same ASIO device for the monitor.",
            },
        )
        self.monitor_device_index = None
        self.monitor_selectors = []
        self.monitor_api = ""

    def start_vc(self, model):
        self.load_model(model)
        # Decoded files are always supplied as mono float32 blocks.  Do not
        # inherit a JACK input endpoint's channel layout for file input.
        self.input_channels = 1 if self.file_source_active else self.get_device_channels()
        self.output_channels = self.get_output_channels()
        self.prepare_inference(self.get_automatic_samplerate(causalvc_backend.SAMPLE_RATE))
        self.status("starting_audio")
        if self.output_stream is not None:
            self.stop()
        self.start_stream()

    def load_model(self, model):
        self.status("loading_model")
        torch.cuda.empty_cache()
        try:
            self.voice = causalvc_backend.CausalVC(
                self.settings.pitch, str(model.model_path), self.config.device
            )
        except Exception as error:
            raise EngineError(
                "model_load_failed",
                f"Could not load {model.model_path.name}: {error}",
                model=model.name,
            ) from error

    def prepare_inference(self, samplerate):
        """Allocate the stream buffers for ``samplerate``; no audio devices."""
        if samplerate != causalvc_backend.SAMPLE_RATE:
            raise EngineError("samplerate_unsupported", "CausalVC voices need a 48 kHz audio route.")
        self.samplerate = samplerate
        self.zc = self.samplerate // 100
        # Whole 20 ms frames; the block itself is the only chunking.
        multiple = causalvc_backend.BLOCK_MULTIPLE
        self.block_frame = max(1, round(self.settings.block_time * self.samplerate / multiple)) * multiple
        self.input_wav = torch.zeros(2 * self.block_frame, device=self.config.device, dtype=torch.float32)
        self.input_wav_denoise = self.input_wav.clone()
        self.rms_buffer = np.zeros(4 * self.zc, dtype="float32")
        # Input denoisers are built on demand for this stream (refresh_denoiser).
        self.active_denoiser = None
        self.active_denoise_delay = 0.0
        self.denoisers = {}
        # Per-stream counters (see __init__).
        self.trimmed_frames = 0
        self.output_underruns = 0
        self.logged_underruns = 0
        self.output_flowing = False
        #: Speech loudness of the output before the gain (output-level indicator).
        self.loudness = LoudnessMeter(self.samplerate)
        self.voice.reset_stream_state()
        self.status("preparing_inference")
        self.voice.warm(self.block_frame)

    def start_stream(self):
        if self.running:
            return
        self.running = True
        self.asio_frames = None
        self.asio_latency = self.asio_request()
        self.log_active_audio_route()
        if self.file_source_active:
            self.start_file_stream()
            return
        if is_native_api(self.input_api):
            self.start_native_input_route()
            return
        self.start_output_stream()
        self.input_stream = sd.InputStream(
            callback=self.watch_asio(self.input_api, self.audio_callback),
            blocksize=self.block_frame,
            samplerate=self.samplerate,
            channels=self.input_channels,
            device=sd.default.device[0],
            dtype="float32",
            **self.stream_kwargs(self.input_api),
        )
        self.input_stream.start()
        self.stream = self.output_stream
        self.start_monitor_stream()

    def start_file_stream(self):
        """Start only the output side; file decoding feeds audio_callback."""
        self.start_output_stream()
        self.stream = self.output_stream
        self.start_monitor_stream()

    def start_output_stream(self, native=None):
        """Open the main output.

        JACK owns its period size, so a JACK output (or any output fed by the
        native input worker) reads from a frame FIFO with blocksize=0.  ALSA
        outputs keep the fixed block queue.
        """
        if native is None:
            native = is_native_api(self.output_api)
        if self.output_api == ASIO_API and self.monitor_shares_asio:
            self.open_asio_stream(with_input=False).start()
            return
        self.output_queue = (
            AudioFrameFifo(1, max_frames=self.block_frame * 4)
            if native
            else queue.Queue(maxsize=3)
        )
        self.output_stream = sd.OutputStream(
            callback=self.watch_asio(self.output_api, self.output_audio_callback),
            blocksize=0 if native else self.block_frame,
            samplerate=self.samplerate,
            channels=self.output_channels,
            device=sd.default.device[1],
            dtype="float32",
            **self.stream_kwargs(self.output_api),
        )
        self.output_stream.start()

    def start_native_input_route(self):
        """Keep JACK's native period size separate from the conversion block."""
        # These are capacity limits, not an added latency preset.  JACK
        # callbacks feed their native 64/128/256-frame periods into the input
        # FIFO; the worker consumes exact blocks.
        fifo_capacity = self.block_frame * 4
        self.native_input_fifo = AudioFrameFifo(
            len(self.input_selectors), max_frames=fifo_capacity
        )
        self.native_worker_stop = threading.Event()
        self.native_worker_wakeup = threading.Event()
        self.native_worker = threading.Thread(
            target=self.native_inference_worker,
            name="causalvc-native-inference",
            daemon=True,
        )
        if self.input_api == ASIO_API and (
            self.output_api == ASIO_API or self.monitor_shares_asio
        ):
            # One full-duplex stream: ASIO cannot open the driver a second time.
            if self.output_api != ASIO_API:
                self.start_output_stream(native=True)
            if not self.monitor_shares_asio:
                self.start_monitor_stream()
            self.open_asio_stream(with_input=True)
            self.native_worker.start()
            self.input_stream.start()
            self.stream = self.input_stream
            return
        self.start_output_stream(native=True)
        self.start_monitor_stream()
        self.input_stream = sd.InputStream(
            callback=self.watch_asio(self.input_api, self.native_fifo_input_callback),
            blocksize=0,
            samplerate=self.samplerate,
            channels=self.input_channels,
            device=sd.default.device[0],
            dtype="float32",
            **self.stream_kwargs(self.input_api),
        )
        self.native_worker.start()
        self.input_stream.start()
        self.stream = self.input_stream

    def open_asio_stream(self, with_input):
        """One stream for every role on the ASIO device: the input (when
        ``with_input``), the main output and the monitor, each on its own
        channels.  Sets the role's stream attributes and queues; not started.
        """
        with_output = self.output_api == ASIO_API
        with_monitor = self.monitor_shares_asio
        device = sd.default.device[1] if with_output else self.monitor_device_index
        selectors = (self.output_selectors if with_output else []) + (
            self.monitor_selectors if with_monitor else []
        )
        out_channels = self.stream_channels(device, "output", selectors)
        if with_output:
            self.output_queue = AudioFrameFifo(1, max_frames=self.block_frame * 4)
            self.output_channels = out_channels
        if with_monitor:
            self.monitor_queue = AudioFrameFifo(1, max_frames=self.block_frame * 4)
            self.monitor_channels = out_channels

        def fill(outdata, frames, times, status):
            if with_output:
                self.output_audio_callback(outdata, frames, times, status)
            else:
                outdata.fill(0)
            if with_monitor:
                self.write_monitor_block(outdata, frames)

        options = self.stream_kwargs(ASIO_API)
        if with_input:
            def duplex(indata, outdata, frames, times, status):
                self.native_fifo_input_callback(indata, frames, times, status)
                fill(outdata, frames, times, status)

            stream = sd.Stream(
                callback=self.watch_asio(ASIO_API, duplex),
                blocksize=0,
                samplerate=self.samplerate,
                channels=(self.input_channels, out_channels),
                device=(sd.default.device[0], device),
                dtype="float32",
                **options,
            )
            self.input_stream = stream
        else:
            stream = sd.OutputStream(
                callback=self.watch_asio(ASIO_API, fill),
                blocksize=0,
                samplerate=self.samplerate,
                channels=out_channels,
                device=device,
                dtype="float32",
                **options,
            )
        if with_output:
            self.output_stream = stream
        if with_monitor:
            self.monitor_stream = stream
        return stream

    # -- ASIO buffer ----------------------------------------------------
    def asio_device(self):
        """The PortAudio index of the ASIO device the selected route uses, or None."""
        settings = self.settings
        candidates = [
            self.devices.outputs.get(settings.output_device),
            self.devices.outputs.get(settings.monitor_device or ""),
        ]
        if not self.file_source_active:
            candidates.insert(0, self.devices.inputs.get(settings.input_device))
        return next(
            (endpoint.index for endpoint in candidates if endpoint and endpoint.api == ASIO_API),
            None,
        )

    def read_asio_sizes(self, device):
        sizes = asio_driver.buffer_sizes(sd, device)
        self.asio_sizes[device] = sizes
        if sizes is not None:
            printt(
                "ASIO driver buffer sizes: min %s, max %s, preferred %s, granularity %s",
                sizes.minimum, sizes.maximum, sizes.preferred, sizes.granularity,
            )
        return sizes

    def asio_state(self):
        device = self.asio_device()
        if device is None:
            return None
        # Reading the sizes loads the driver, which an open stream holds
        # (file preview holds an output without running).
        if device not in self.asio_sizes and not self.streams_open():
            self.read_asio_sizes(device)
        sizes = self.asio_sizes.get(device)
        return {
            "sizes": sizes.to_dict() if sizes else None,
            "in_use": self.asio_frames if self.running else None,
        }

    def asio_request(self):
        """The latency to ask ASIO streams for: the chosen buffer, or the
        driver's current size (read now, so a change made in its control panel
        counts without reloading devices)."""
        device = self.asio_device()
        if device is None:
            return None
        sizes = self.read_asio_sizes(device)
        frames = self.settings.asio_buffer or (sizes.preferred if sizes else 0)
        if not frames:
            return None
        if sizes is not None and self.settings.asio_buffer:
            opened = sizes.accepts(frames)
            if opened != frames:
                printt("The ASIO driver does not take %s samples; it opens %s.", frames, opened)
        return frames / self.samplerate

    def streams_open(self):
        return self.running or any(
            getattr(self, name, None) is not None
            for name in ("input_stream", "output_stream", "monitor_stream")
        )

    def stream_kwargs(self, api):
        return stream_options(sd, api, self.asio_latency if api == ASIO_API else None)

    def watch_asio(self, api, callback):
        """Note the buffer size an ASIO stream's callbacks receive."""
        if api != ASIO_API:
            return callback

        def watched(*args):
            self.asio_frames = args[-3]  # (..., frames, time, status)
            return callback(*args)

        return watched

    def open_asio_control_panel(self):
        device = self.asio_device()
        if device is None:
            raise EngineError("bad_request", "No ASIO device is selected.")
        if self.streams_open():
            raise EngineError("asio_panel_busy", "Stop the voice changer before opening the driver's control panel.")
        if not asio_driver.show_control_panel(sd, device):
            raise EngineError("asio_panel_failed", "The ASIO driver's control panel could not be opened.")
        # The panel may have changed the buffer size.
        self.asio_sizes.pop(device, None)
        self.emit_state()

    def native_inference_worker(self):
        while not self.native_worker_stop.is_set():
            self.native_worker_wakeup.wait(0.05)
            self.native_worker_wakeup.clear()
            while not self.native_worker_stop.is_set():
                input_block = self.native_input_fifo.read(self.block_frame, exact=True)
                if input_block is None:
                    break
                try:
                    self.audio_callback(input_block, self.block_frame, None, None)
                except Exception:
                    printt(traceback.format_exc())
                    self.native_worker_stop.set()
                    return

    def native_fifo_input_callback(self, indata, frames, times, status):
        self.native_input_fifo.write(select_channels(indata, self.input_selectors))
        self.native_worker_wakeup.set()

    def output_audio_callback(self, outdata, frames, times, status):
        block = self.read_audio_target(self.output_queue, frames)
        if block is not None:
            self.output_flowing = True
        if self.output_flowing and self.running and (block is None or block.shape[0] < frames):
            self.output_underruns += 1
        self.write_output_block(outdata, block)
        # Use the real device callback timeline for recording.  The output
        # can be delayed by the engine's buffers, which is intentionally preserved.
        if block is None:
            self.recorder.enqueue("output", np.zeros(frames, dtype=np.float32))
        else:
            self.recorder.enqueue("output", block[:, 0])

    @staticmethod
    def read_audio_target(target, frames):
        if target is None:
            return None
        if isinstance(target, AudioFrameFifo):
            return target.read(frames)
        try:
            block = target.get_nowait()
        except queue.Empty:
            return None
        if block.ndim == 1:
            block = block[:, None]
        return block

    def write_output_block(self, outdata, block):
        outdata.fill(0)
        if block is None:
            return
        sample_count = min(outdata.shape[0], block.shape[0])
        mono = block[:sample_count, 0] if block.ndim == 2 else block[:sample_count]
        scatter_mono(outdata, mono, self.output_selectors)

    def start_monitor_stream(self):
        if self.monitor_shares_asio:
            return  # opened with the ASIO stream it shares
        self.monitor_stream = None
        if self.monitor_device_index is None:
            self.monitor_queue = None
            return
        try:
            self.monitor_channels = self.stream_channels(
                self.monitor_device_index, "output", self.monitor_selectors
            )
            monitor_options = self.stream_kwargs(self.monitor_api)
            sd.check_output_settings(
                device=self.monitor_device_index,
                channels=self.monitor_channels,
                dtype="float32",
                samplerate=self.samplerate,
                **check_options(monitor_options),
            )
            self.monitor_queue = AudioFrameFifo(1, max_frames=self.block_frame * 4)
            self.monitor_stream = sd.OutputStream(
                device=self.monitor_device_index,
                callback=self.watch_asio(self.monitor_api, self.monitor_audio_callback),
                blocksize=0,
                samplerate=self.samplerate,
                channels=self.monitor_channels,
                dtype="float32",
                **monitor_options,
            )
            self.monitor_stream.start()
        except Exception as error:
            if self.monitor_stream is not None:
                self.monitor_stream.close()
            self.monitor_stream = None
            self.monitor_queue = None
            # Not fatal: conversion continues on the main output only.
            self.emit("error", {"code": "monitor_failed", "message": str(error)})

    def monitor_audio_callback(self, outdata, frames, times, status):
        outdata.fill(0)
        self.write_monitor_block(outdata, frames)

    def write_monitor_block(self, outdata, frames):
        """Write the next monitor audio to its channels, leaving the others."""
        block = self.read_audio_target(self.monitor_queue, frames)
        if block is None:
            return
        sample_count = min(frames, block.shape[0])
        mono = block[:sample_count, 0] if block.ndim == 2 else block[:sample_count]
        scatter_mono(outdata, mono, self.monitor_selectors)

    def stop(self):
        was_running = self.running
        self.running = False
        self.stop_file_playback()
        if self.recorder.active:
            self.stop_recording()
        if self.native_worker_stop is not None:
            self.native_worker_stop.set()
        if self.native_worker_wakeup is not None:
            self.native_worker_wakeup.set()
        self.close_streams("input_stream", "output_stream", "monitor_stream")
        self.stream = None
        self.asio_frames = None
        self.output_queue = None
        self.monitor_queue = None
        self.native_input_fifo = None
        if self.native_worker is not None and self.native_worker.is_alive():
            self.native_worker.join(timeout=0.5)
        self.native_worker = None
        self.native_worker_stop = None
        self.native_worker_wakeup = None
        self.latest_input_meter = 0.0
        self.latest_output_meter = 0.0
        self.latest_monitor_meter = 0.0
        if was_running:
            self.status(
                "passthrough_stopped" if self.function == "im" else "conversion_stopped"
            )
            self.emit_state()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def audio_callback(self, indata, frames, times, status):
        """Process one input-audio callback block."""
        start_time = time.perf_counter()
        settings = self.settings
        indata = librosa.to_mono(indata.T)
        indata *= np.float32(settings.input_gain)
        np.clip(indata, -1.0, 1.0, out=indata)
        self.recorder.enqueue("input", indata)
        meter_now = time.perf_counter()
        if meter_now - self.last_input_meter_update >= METER_INTERVAL_SECONDS:
            self.latest_input_meter = peak_meter(indata)
            self.last_input_meter_update = meter_now
        denoised = self.ingest_block(indata)
        block = (self.input_wav_denoise if denoised else self.input_wav)[-self.block_frame :]
        # Every block is converted, silence included: the model's own state has to keep running.
        output = self.voice.process(block) if self.function == "vc" else block.clone()
        return self.finish_block(output, start_time)

    # ------------------------------------------------------------------
    # Input noise reduction

    def hushwire_status(self):
        """For the UI: "ready", "needs_download", "downloading", "failed" or "unavailable"."""
        if not hushwire.available():
            return "unavailable"
        if hushwire.model_ready():
            return "ready"
        return self.hushwire_download or "needs_download"

    def build_denoiser(self, method):
        """A fresh denoiser and its delay in seconds, or None when it can't run here."""
        if method == "rnnoise":
            if not rnnoise.available():
                return None
            return rnnoise.RNNoise(), RNNOISE_DELAY_SECONDS
        if method == "hushwire" and hushwire.available() and hushwire.model_ready():
            try:
                denoiser = hushwire.Hushwire()
            except Exception as error:  # a damaged model file, or the runtime refusing it
                printt("Hushwire could not start: %s", error)
                return None
            return denoiser, denoiser.delay_seconds(self.block_frame)
        return None

    def refresh_denoiser(self):
        """Use the input denoiser the settings ask for, keeping the reported delay in step."""
        with self.denoise_lock:
            if not self.running or getattr(self, "block_frame", None) is None:
                return
            method = self.settings.input_denoise_method
            chosen = None
            if self.settings.input_denoise:
                if method not in self.denoisers:
                    self.denoisers[method] = self.build_denoiser(method)
                chosen = self.denoisers[method]
            denoiser, delay = chosen or (None, 0.0)
            if denoiser is self.active_denoiser:
                return
            self.delay_time += delay - self.active_denoise_delay
            self.active_denoiser, self.active_denoise_delay = denoiser, delay
            names = {"rnnoise": "RNNoise", "hushwire": "Hushwire (Amazon Voice Focus)"}
            if denoiser is not None:
                printt("Input noise reduction: %s, %d ms", names[method], round(delay * 1000))
            elif self.settings.input_denoise:
                printt("Input noise reduction: %s is not available, so the input is not denoised", names[method])
            else:
                printt("Input noise reduction: off")

    def download_hushwire(self):
        """Accept Hushwire's download notice and fetch its model in the background."""
        if not hushwire.available():
            raise EngineError("hushwire_unavailable", "wasmtime is not installed; reinstall the project.")
        if hushwire.model_ready() or self.hushwire_download == "downloading":
            return
        hushwire.accept_download()
        self.hushwire_download = "downloading"
        self.emit_state()

        def fetch():
            try:
                path = hushwire.ensure_model()
            except Exception as error:
                self.hushwire_download = "failed"
                printt("Hushwire model download failed: %s", error)
                self.emit("error", {"code": "hushwire_download_failed", "message": str(error)})
            else:
                self.hushwire_download = None
                printt("Hushwire model downloaded: %s", path)
                self.status("hushwire_downloaded")
                with self.denoise_lock:
                    # A denoiser that was missing for want of the model is built now.
                    if self.denoisers.get("hushwire", False) is None:
                        del self.denoisers["hushwire"]
                    self.refresh_denoiser()
            self.emit_state()

        threading.Thread(target=fetch, name="hushwire-download", daemon=True).start()

    def ingest_block(self, indata):
        """Write one input block into the buffers: noise gate, then denoise.

        Returns whether the denoised buffer holds this block.
        """
        settings = self.settings
        if settings.noise_gate_db > -60:
            indata = np.append(self.rms_buffer, indata)
            rms = librosa.feature.rms(
                y=indata, frame_length=4 * self.zc, hop_length=self.zc
            )[:, 2:]
            self.rms_buffer[:] = indata[-4 * self.zc :]
            indata = indata[2 * self.zc - self.zc // 2 :]
            db_threhold = (
                librosa.amplitude_to_db(rms, ref=1.0)[0] < settings.noise_gate_db
            )
            for i in range(db_threhold.shape[0]):
                if db_threhold[i]:
                    indata[i * self.zc : (i + 1) * self.zc] = 0
            indata = indata[self.zc // 2 :]
        self.input_wav[: -self.block_frame] = self.input_wav[self.block_frame :].clone()
        self.input_wav[-indata.shape[0] :] = torch.from_numpy(indata).to(self.config.device)
        denoiser = self.active_denoiser
        if denoiser is None:
            return False
        # Both denoisers keep a recurrent state, so they see each block once, in order.
        self.input_wav_denoise[: -self.block_frame] = self.input_wav_denoise[self.block_frame :].clone()
        denoised = denoiser.process(indata)
        self.input_wav_denoise[-self.block_frame :] = torch.from_numpy(denoised).to(self.config.device)
        return True

    def finish_block(self, output, start_time):
        """Apply output gain and file volume, update meters, queue the block."""
        settings = self.settings
        before_gain = output[: self.block_frame].cpu().numpy()
        self.loudness.process(before_gain)
        # Soft clip rather than a hard clamp: loud peaks are rounded, not chopped.
        output_mono = soft_clip(before_gain * np.float32(settings.output_gain))
        # The file-player volume is applied after conversion, so each 1% step controls the audible result.
        if self.file_source_active:
            output_mono *= np.float32(np.clip(settings.file_input_volume, 0.0, 1.0))
        monitor_mono = soft_clip(output_mono * np.float32(settings.monitor_gain))
        meter_now = time.perf_counter()
        if meter_now - self.last_output_meter_update >= METER_INTERVAL_SECONDS:
            self.latest_output_meter = peak_meter(output_mono)
            self.latest_monitor_meter = (
                0.0 if self.monitor_device_index is None else peak_meter(monitor_mono)
            )
            self.last_output_meter_update = meter_now
        self.queue_output(self.output_queue, output_mono)
        self.queue_output(self.monitor_queue, monitor_mono)
        if self.running:
            self.latest_infer_time = int((time.perf_counter() - start_time) * 1000)
        return output_mono

    def queued_frames(self, target):
        """Frames waiting in an output queue or FIFO."""
        if target is None:
            return 0
        if isinstance(target, AudioFrameFifo):
            return target.available_frames
        return target.qsize() * self.block_frame

    def queue_output(self, target, block):
        """Queue a finished block, or drop it if it is silence over a backlog."""
        if target is None:
            return
        if not self.file_source_active and not np.any(block):
            margin = max(BACKLOG_MARGIN_SECONDS, 2 * self.latest_infer_time / 1000)
            queued = self.queued_frames(target)
            if queued >= self.block_frame + int(margin * self.samplerate):
                self.trimmed_frames += block.shape[0]
                printt(
                    "Output backlog %.0f ms: dropped %.0f ms of silence to catch up",
                    1000 * queued / self.samplerate,
                    1000 * block.shape[0] / self.samplerate,
                )
                return
        self.enqueue_audio_target(target, block)

    def log_underruns(self):
        """Command thread: report new output underruns every few seconds."""
        new = self.output_underruns - self.logged_underruns
        now = time.monotonic()
        if new <= 0 or now - self.underrun_logged_at < UNDERRUN_LOG_SECONDS:
            return
        self.logged_underruns = self.output_underruns
        self.underrun_logged_at = now
        printt("Output underruns: %d (+%d): the output ran dry, an audible gap", self.output_underruns, new)

    def queued_ms(self):
        """Audio waiting between the mic and the output right now, in ms."""
        if not self.running or not self.samplerate:
            return None
        frames = self.queued_frames(self.output_queue)
        if self.native_input_fifo is not None:
            frames += self.native_input_fifo.available_frames
        return round(1000 * frames / self.samplerate)

    @staticmethod
    def enqueue_audio_target(target, block):
        if target is None:
            return
        if isinstance(target, AudioFrameFifo):
            target.write(block)
        else:
            enqueue_latest(target, block.copy())

    # ------------------------------------------------------------------
    # Routing helpers
    # ------------------------------------------------------------------
    def get_automatic_samplerate(self, model_rate):
        """The active audio route's clock; the model itself needs 48 kHz (see prepare_inference)."""
        return self.get_routing_samplerate(preferred_rates=[model_rate])

    def get_routing_samplerate(self, preferred_rates=()):
        """Find a sample rate that every active stream can actually open."""
        input_info = sd.query_devices(device=sd.default.device[0])
        output_info = sd.query_devices(device=sd.default.device[1])

        candidates = []
        # The model's rate first: any other rate fails in prepare_inference
        # anyway, and WASAPI converts to it when the mixer runs at another.
        for rate in (
            *preferred_rates,
            *(() if self.file_source_active else (input_info["default_samplerate"],)),
            output_info["default_samplerate"],
            48000,
            44100,
            40000,
        ):
            rate = int(round(rate))
            if rate > 0 and rate not in candidates:
                candidates.append(rate)

        checks = [
            (sd.check_output_settings, sd.default.device[1], self.output_channels, self.output_api)
        ]
        if not self.file_source_active:
            checks.append(
                (sd.check_input_settings, sd.default.device[0], self.input_channels, self.input_api)
            )
        if self.monitor_device_index is not None:
            checks.append(
                (
                    sd.check_output_settings,
                    self.monitor_device_index,
                    self.stream_channels(
                        self.monitor_device_index, "output", self.monitor_selectors
                    ),
                    self.monitor_api,
                )
            )

        errors = []
        for rate in candidates:
            try:
                for checker, device, channels, api in checks:
                    checker(
                        device=device,
                        channels=channels,
                        dtype="float32",
                        samplerate=rate,
                        **check_options(stream_options(sd, api)),
                    )
            except sd.PortAudioError as error:
                errors.append(f"{rate} Hz: {error}")
                continue
            printt("Selected common sample rate: %s", rate)
            return rate

        raise EngineError(
            "no_common_samplerate",
            "The selected input, output, and monitor devices have no common sample rate. "
            "Set the devices to the same rate (usually 48 kHz or 44.1 kHz).\n"
            + "\n".join(errors[-3:]),
        )

    @staticmethod
    def stream_channels(device, direction, selectors):
        """Channels to open: enough to reach every selected JACK port."""
        if selectors:
            return max(selectors) + 1
        max_key = "max_input_channels" if direction == "input" else "max_output_channels"
        return min(int(sd.query_devices(device=device)[max_key]), 2)

    def get_device_channels(self):
        return self.stream_channels(sd.default.device[0], "input", self.input_selectors)

    def get_output_channels(self):
        return self.stream_channels(sd.default.device[1], "output", self.output_selectors)

    # ------------------------------------------------------------------
    # File input
    # ------------------------------------------------------------------
    def file_select(self, path):
        if not isinstance(path, str) or not os.path.isfile(path):
            raise EngineError("no_audio_file", f"Audio file not found: {path}")
        if not os.path.isfile(self.ffmpeg_path):
            raise EngineError(
                "ffmpeg_missing",
                "FFmpeg was not found. Install ffmpeg and make sure it is on PATH.",
            )
        self.stop_file_playback()
        self.file_input_path = path
        self.status("file_selected", name=os.path.basename(path))
        self.emit_state()

    def file_play(self):
        if not self.file_source_active or not self.file_input_path:
            raise EngineError("no_audio_file", "Select an audio file first.")
        if not os.path.isfile(self.ffmpeg_path):
            raise EngineError("ffmpeg_missing", "FFmpeg was not found.")
        if self.output_stream is None:
            try:
                self.prepare_file_playback_output()
            except EngineError:
                raise
            except Exception as error:
                raise EngineError("audio_start_failed", str(error)) from error
        if self.file_audio_source is None:
            settings = self.settings
            # Blocks of silence fed after the file ends, so the model's delay plays out.
            tail_blocks = max(2, int(0.1 / settings.block_time) + 2)
            try:
                self.file_audio_source = FileAudioSource(
                    self.ffmpeg_path,
                    self.file_input_path,
                    self.samplerate,
                    self.block_frame,
                    self.handle_file_audio_block,
                    tail_blocks=tail_blocks,
                )
            except OSError as error:
                raise EngineError("file_error", str(error)) from error
        self.file_audio_source.play()
        self.status("playing")
        self.emit_state()

    def file_pause(self):
        if self.file_audio_source is not None:
            self.file_audio_source.pause()
        self.status("paused")
        self.emit_state()

    def file_stop(self):
        self.stop_file_playback()
        self.status("playback_stopped")
        self.emit_state()

    def file_seek(self, seconds):
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
            raise EngineError("bad_request", "seconds must be a number")
        if self.file_audio_source is not None:
            self.file_audio_source.seek(max(0.0, float(seconds)))

    def prepare_file_playback_output(self):
        """Open an output-only stream to preview a file without converting."""
        self.set_devices()
        self.input_channels = 1
        self.output_channels = self.get_output_channels()
        self.samplerate = self.get_automatic_samplerate(48000)
        self.zc = self.samplerate // 100
        self.asio_frames = None
        self.asio_latency = self.asio_request()
        self.block_frame = max(
            self.zc,
            int(round(self.settings.block_time * self.samplerate / self.zc)) * self.zc,
        )
        self.start_file_stream()

    def handle_file_audio_block(self, block):
        if self.running:
            self.audio_callback(block[:, None], self.block_frame, None, None)
            return
        volume = float(np.clip(self.settings.file_input_volume, 0.0, 1.0))
        block = soft_clip(block * np.float32(volume))
        self.enqueue_audio_target(self.output_queue, block)
        self.enqueue_audio_target(
            self.monitor_queue, soft_clip(block * np.float32(self.settings.monitor_gain))
        )

    def stop_file_playback(self):
        if self.file_audio_source is not None:
            self.file_audio_source.stop()
            self.file_audio_source = None
        self.file_state = (False, False)
        if not self.running and self.output_stream is not None:
            # Close the preview-only output opened by file_play.
            self.close_streams("output_stream", "monitor_stream")
            self.stream = None

    def close_streams(self, *names):
        """Close the named streams once each; a shared ASIO stream is held by
        several roles, and every role holding it is cleared."""
        roles = ("input_stream", "output_stream", "monitor_stream")
        for name in names:
            stream = getattr(self, name, None)
            if stream is None:
                continue
            stream.abort()
            stream.close()
            for role in roles:
                if getattr(self, role, None) is stream:
                    setattr(self, role, None)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def start_recording(self):
        if not self.running:
            raise EngineError("record_requires_running", "Start conversion before recording.")
        if self.recorder.active:
            return
        folder = self.settings.recording_folder or DEFAULT_RECORDING_FOLDER
        try:
            self.recorder.start(folder, self.samplerate, self.settings.recording_mode)
        except OSError as error:
            raise EngineError("recording_failed", str(error)) from error
        self.status("recording_started")
        self.emit_state()

    def stop_recording(self):
        if not self.recorder.active:
            return []
        self.recorder.stop()
        saved_paths = list(getattr(self.recorder, "last_saved_paths", []))
        if saved_paths:
            self.status("recording_saved", paths=saved_paths)
        self.emit_state()
        return saved_paths
