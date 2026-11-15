"""Hushwire noise suppression (Amazon Voice Focus) for the engine.

A Python port of Hushwire's ``vf-core`` (https://github.com/mojowozo/Hushwire):
the model is a self-contained WebAssembly module, run here with wasmtime and
driven in 128-frame quanta exactly like the Chime SDK's AudioWorklet does.

The model is Amazon's proprietary software and is never bundled. It is fetched
from Amazon's CDN only after the user accepts ``DOWNLOAD_NOTICE``, and is kept
in Hushwire's own folder (``~/.local/share/Hushwire/models`` or
``%APPDATA%\\Hushwire\\models``), so the plugin and this engine share one copy
and one acceptance.
"""

import os
import sys
import threading
import urllib.request
from pathlib import Path

import numpy as np

try:
    import wasmtime
except ImportError:  # the engine still runs; Hushwire just reports itself unavailable
    wasmtime = None

SAMPLE_RATE = 48000
#: The SDK feeds the model in Web Audio render quanta of this many frames.
QUANTUM = 128
#: Internal model chunk: 10 ms at 48 kHz.
PROCESS_FRAMES = 480
#: Hushwire's default model; the variants run c10 (lightest) to c100.
DEFAULT_VARIANT = "c20"
VARIANTS = ("c10", "c20", "c50", "c100")
#: The network's own delay at 48 kHz on top of the buffering path (vf-core's MODEL_DELAY_48K).
MODEL_DELAY_FRAMES = 1499
CDN = "https://static.sdkassets.chime.aws/wasm"
ASSET_GROUP = "stable-v1"
#: AGC target level the SDK passes to create_processor (unused while AGC is off).
AGC_LEVEL = -15
MAX_DOWNLOAD_BYTES = 64 << 20

#: What the apps show before the first download (hushwire-notice-body); the user must accept it.
#: It is written into the acceptance marker, which Hushwire's plugin shares.
DOWNLOAD_NOTICE = (
    "Hushwire's noise reduction model is Amazon Voice Focus, which is Amazon's proprietary software and is "
    "not included with CausalVC. Downloading fetches it from Amazon's servers (static.sdkassets.chime.aws) "
    "to this computer. Its use is subject to Amazon's terms. Neither Hushwire nor CausalVC is affiliated "
    "with or endorsed by Amazon."
)


def available():
    """Whether the WebAssembly runtime is installed."""
    return wasmtime is not None


def data_dir():
    """Hushwire's per-user folder, where the Rust ``dirs::data_dir()`` puts it."""
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming") / "Hushwire"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Hushwire"
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "Hushwire"


def model_path(variant=DEFAULT_VARIANT):
    return data_dir() / "models" / f"voicefocus-default-{variant}-v1_simd.wasm"


def model_ready(variant=DEFAULT_VARIANT):
    return model_path(variant).is_file()


def _consent_marker():
    return data_dir() / "accepted-download-notice"


def download_accepted():
    return _consent_marker().exists()


def accept_download():
    """Record that the user accepted ``DOWNLOAD_NOTICE`` (the same marker Hushwire writes)."""
    marker = _consent_marker()
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(DOWNLOAD_NOTICE + "\n", encoding="utf-8")


class DownloadNotAccepted(RuntimeError):
    def __init__(self):
        super().__init__("the model download notice hasn't been accepted yet")


def ensure_model(variant=DEFAULT_VARIANT, timeout=60):
    """The cached model's path, downloading it first if needed (and accepted)."""
    path = model_path(variant)
    if path.is_file():
        return path
    if not download_accepted():
        raise DownloadNotAccepted()
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{CDN}/{path.name}?assetGroup={ASSET_GROUP}"
    with urllib.request.urlopen(url, timeout=timeout) as response:
        content_type = response.headers.get("content-type", "")
        if content_type != "application/wasm":
            raise RuntimeError(f"unexpected content type {content_type!r} from {url}")
        data = response.read(MAX_DOWNLOAD_BYTES + 1)
    if len(data) > MAX_DOWNLOAD_BYTES or not data.startswith(b"\0asm"):
        raise RuntimeError(f"download from {url} is not a WASM module")
    part = path.with_name(path.name + ".part")
    part.write_bytes(data)
    os.replace(part, path)
    return path


_compiled = {}
_compiled_lock = threading.Lock()


def _module(path):
    """Compile once per model file; instances share it."""
    key = (str(path), path.stat().st_mtime_ns)
    with _compiled_lock:
        if key not in _compiled:
            config = wasmtime.Config()
            config.cranelift_opt_level = "speed"
            engine = wasmtime.Engine(config)
            _compiled[key] = (engine, wasmtime.Module.from_file(engine, str(path)))
        return _compiled[key]


class _VoiceFocus:
    """One mono 48 kHz model instance, one 128-frame quantum at a time (vf-core's VoiceFocus)."""

    def __init__(self, path):
        engine, module = _module(path)
        self.store = wasmtime.Store(engine)
        linker = wasmtime.Linker(engine)
        # The model only imports WASI stubs; the SDK answers them with no-ops.
        for item in module.imports:
            results = item.type.results
            linker.define_func(item.module, item.name, item.type, (lambda *_, r=results: 0 if r else None))
        exports = linker.instantiate(self.store, module).exports(self.store)
        self.memory = exports["memory"]
        self._step = exports["step"]
        self._provide = exports["provide_chunk"]
        self._receive = exports["receive_chunk"]
        self._process = exports["process_fullband_chunk"]
        self.buffer = exports["create_audio_buffer"](self.store, PROCESS_FRAMES, 4, 8)
        if not self.buffer:
            raise RuntimeError("the model refused the 48 kHz buffer")
        self.processor = exports["create_processor"](self.store, 0, AGC_LEVEL)
        if not self.processor:
            raise RuntimeError("create_processor failed")
        malloc = exports["malloc"]
        self.in_ptr = malloc(self.store, QUANTUM * 4)
        self.out_ptr = malloc(self.store, QUANTUM * 4)
        self.chunk_ptr = malloc(self.store, PROCESS_FRAMES * 4)
        if not (self.in_ptr and self.out_ptr and self.chunk_ptr):
            raise RuntimeError("model malloc failed")
        #: When False, audio still passes through the model's buffering (same latency) but isn't cleaned.
        self.enabled = True

    def quantum(self, samples):
        """Process 128 float32 samples; returns the 128 the model gives back."""
        store = self.store
        self.memory.write(store, samples.astype("<f4", copy=False).tobytes(), self.in_ptr)
        self._step(store, self.buffer, QUANTUM, self.out_ptr, self.in_ptr)
        output = np.frombuffer(self.memory.read(store, self.out_ptr, self.out_ptr + QUANTUM * 4), dtype="<f4")
        if self._provide(store, self.buffer, self.chunk_ptr) == 1:
            if self.enabled:
                self._process(store, self.processor, self.chunk_ptr)
            self._receive(store, self.buffer, self.chunk_ptr)
        return output


def _buffer_delay(path):
    """Frames of the model's buffering path, measured with an impulse (vf-core's latency_samples)."""
    model = _VoiceFocus(path)
    model.enabled = False
    at = SAMPLE_RATE // 5  # after the buffers have settled; the first chunks behave differently
    signal = np.zeros(2 * at, dtype=np.float32)
    signal[at] = 1.0
    output = np.concatenate([model.quantum(signal[i : i + QUANTUM]) for i in range(0, signal.shape[0], QUANTUM)])
    lag = int(np.argmax(np.abs(output[at:])))
    if abs(output[at + lag]) < 0.1:
        raise RuntimeError("no impulse through the model's buffer path")
    return lag


class Hushwire:
    """Streaming noise suppression for consecutive 48 kHz blocks of any length.

    Voice Focus outputs inverted polarity; ``process`` corrects it, so the
    result lines up with the dry signal apart from ``delay_frames``.
    """

    def __init__(self, variant=DEFAULT_VARIANT):
        if wasmtime is None:
            raise RuntimeError("wasmtime is not installed")
        path = model_path(variant)
        if not path.is_file():
            raise FileNotFoundError(f"Hushwire model {variant} has not been downloaded")
        self.variant = variant
        self._model = _VoiceFocus(path)
        self._pending = np.zeros(0, dtype=np.float32)
        #: Processed samples not yet handed out; starts with one quantum of zeros so any
        #: block size can be served (set to empty on the first call when blocks are whole quanta).
        self._ready = None
        self._buffer_frames = _buffer_delay(path)
        self.delay_frames = None

    def delay_seconds(self, block_frames):
        """Total delay for blocks of ``block_frames`` samples."""
        pad = 0 if block_frames % QUANTUM == 0 else QUANTUM
        return (pad + self._buffer_frames + MODEL_DELAY_FRAMES) / SAMPLE_RATE

    def process(self, audio):
        audio = np.asarray(audio, dtype=np.float32)
        if self._ready is None:
            pad = 0 if audio.shape[0] % QUANTUM == 0 else QUANTUM
            self._ready = np.zeros(pad, dtype=np.float32)
            self.delay_frames = pad + self._buffer_frames + MODEL_DELAY_FRAMES
        pending = np.concatenate([self._pending, audio])
        whole = pending.shape[0] - pending.shape[0] % QUANTUM
        cleaned = [self._model.quantum(pending[i : i + QUANTUM]) for i in range(0, whole, QUANTUM)]
        self._pending = pending[whole:]
        ready = np.concatenate([self._ready, *cleaned])
        if ready.shape[0] < audio.shape[0]:  # only if the block size changed mid-stream
            ready = np.concatenate([np.zeros(audio.shape[0] - ready.shape[0], dtype=np.float32), ready])
        self._ready = ready[audio.shape[0] :]
        return -ready[: audio.shape[0]]
