"""Engine settings, defaults, and their JSON persistence.

Global settings live in ``configs/engine.json``.  The per-model settings
(gains, noise gate, pitch) live beside each voice in ``realtime_settings.json``.
"""

import json
import math
import os
from dataclasses import asdict, dataclass, fields

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "engine.json")
DEFAULT_RECORDING_FOLDER = os.path.join(PROJECT_ROOT, "recordings")

INPUT_SOURCES = ("microphone", "file")
RECORDING_MODES = ("separate", "mix", "stereo")
#: Output-level indicator targets (tools/loudness.py TARGETS_LUFS).
LOUDNESS_TARGETS = ("voice_chat", "streaming")
#: Input noise reduction: Hushwire (Amazon Voice Focus, ~53 ms) or RNNoise (20 ms).
INPUT_DENOISE_METHODS = ("hushwire", "rnnoise")
#: ASIO buffer sizes (samples) the app can ask the driver for; 0 keeps the driver's own setting.
ASIO_BUFFER_SIZES = (0, 32, 64, 96, 128, 192, 256, 512, 1024, 2048)

MODEL_SETTINGS_FILENAME = "realtime_settings.json"
# Protocol name -> key stored in each model's realtime_settings.json.
MODEL_SETTING_FILE_KEYS = {
    "input_gain_db": "input_gain_db",
    "output_gain_db": "output_gain_db",
    "monitor_gain_db": "monitor_gain_db",
    "noise_gate_db": "threhold",
    "pitch": "pitch",
}
MODEL_SETTING_KEYS = tuple(MODEL_SETTING_FILE_KEYS)
RESETTABLE_MODEL_SETTING_KEYS = MODEL_SETTING_KEYS

# Settings that only matter when the next stream starts; changing one stops
# a running stream.
RESTART_SETTING_KEYS = frozenset(
    (
        "model_name",
        "gpu",
        "input_source",
        "input_device",
        "output_device",
        "monitor_device",
        "block_time",
        "asio_buffer",
    )
)


@dataclass
class EngineSettings:
    # Global settings (engine.json).
    model_name: str = ""
    gpu: str = "auto"
    input_source: str = "microphone"
    input_device: str = ""
    output_device: str = ""
    monitor_device: str | None = None
    #: Seconds of audio converted at a time; a multiple of 20 ms. Delay is the model's 60 ms plus one block.
    block_time: float = 0.04
    #: ASIO buffer size to ask the driver for, in samples; 0 keeps the driver's own setting.
    asio_buffer: int = 0
    input_denoise: bool = False
    input_denoise_method: str = "hushwire"
    #: What the output-level indicator aims for: voice chat (-21 LUFS) or
    #: streaming (-16 LUFS).
    loudness_target: str = "voice_chat"
    recording_folder: str = DEFAULT_RECORDING_FOLDER
    recording_mode: str = "separate"
    file_input_volume: float = 1.0
    # Per-model settings (realtime_settings.json beside the model).
    input_gain_db: float = 0.0
    output_gain_db: float = 0.0
    monitor_gain_db: float = 0.0
    noise_gate_db: float = -60.0
    pitch: float = 0.0

    @property
    def input_gain(self):
        return db_to_linear(self.input_gain_db)

    @property
    def output_gain(self):
        return db_to_linear(self.output_gain_db)

    @property
    def monitor_gain(self):
        return db_to_linear(self.monitor_gain_db)

    def to_dict(self):
        return asdict(self)

    def global_dict(self):
        return {
            key: value
            for key, value in asdict(self).items()
            if key not in MODEL_SETTING_KEYS
        }

    def model_dict(self):
        return {key: getattr(self, key) for key in MODEL_SETTING_KEYS}


SETTING_TYPES = {field.name: field.type for field in fields(EngineSettings)}
PERFORMANCE_DEFAULT_KEYS = (
    "block_time",
    "input_denoise",
    "input_denoise_method",
)


def db_to_linear(db):
    return 10.0 ** (float(db) / 20.0)


class SettingError(ValueError):
    def __init__(self, key, message):
        super().__init__(message)
        self.key = key


def coerce_setting(key, value):
    """Validate one protocol value and convert it to the stored type."""
    if key not in SETTING_TYPES:
        raise SettingError(key, f"Unknown setting: {key}")
    kind = SETTING_TYPES[key]
    if kind is int:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value != int(value):
            raise SettingError(key, f"{key} must be a whole number")
        value = int(value)
        if key == "asio_buffer" and value not in ASIO_BUFFER_SIZES:
            raise SettingError(key, f"asio_buffer must be one of {', '.join(map(str, ASIO_BUFFER_SIZES))}")
        return value
    if kind is bool:
        if not isinstance(value, bool):
            raise SettingError(key, f"{key} must be true or false")
        return value
    if kind is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SettingError(key, f"{key} must be a number")
        value = float(value)
        if not math.isfinite(value):
            raise SettingError(key, f"{key} must be finite")
        if key in ("block_time",):
            value = max(0.02, value)
        if key == "file_input_volume":
            value = min(max(value, 0.0), 1.0)
        return value
    if key == "monitor_device":
        if value is not None and not isinstance(value, str):
            raise SettingError(key, "monitor_device must be a string or null")
        return value or None
    if not isinstance(value, str):
        raise SettingError(key, f"{key} must be a string")
    allowed = {
        "input_source": INPUT_SOURCES,
        "recording_mode": RECORDING_MODES,
        "loudness_target": LOUDNESS_TARGETS,
        "input_denoise_method": INPUT_DENOISE_METHODS,
    }.get(key)
    if allowed and value not in allowed:
        raise SettingError(key, f"{key} must be one of {', '.join(allowed)}")
    return value


def load_settings(path=ENGINE_CONFIG_PATH):
    """Read engine.json; invalid or unknown entries fall back to defaults."""
    settings = EngineSettings()
    try:
        with open(path, "r", encoding="utf-8") as config_file:
            saved = json.load(config_file)
    except (OSError, ValueError):
        return settings
    if not isinstance(saved, dict):
        return settings
    for key, value in saved.items():
        if key in MODEL_SETTING_KEYS:
            continue
        try:
            setattr(settings, key, coerce_setting(key, value))
        except SettingError:
            continue
    return settings


def save_settings(settings, path=ENGINE_CONFIG_PATH):
    write_json_atomic(path, settings.global_dict())


def load_model_settings(directory):
    """Return one model's per-model settings, or the defaults."""
    defaults = EngineSettings()
    values = {key: getattr(defaults, key) for key in MODEL_SETTING_KEYS}
    if directory is None:
        return values
    path = os.path.join(directory, MODEL_SETTINGS_FILENAME)
    try:
        with open(path, "r", encoding="utf-8") as settings_file:
            saved = json.load(settings_file)
    except FileNotFoundError:
        return values
    except (OSError, ValueError):
        print(f"Could not read model settings: {path}")
        return values
    if not isinstance(saved, dict):
        return values
    for key, file_key in MODEL_SETTING_FILE_KEYS.items():
        value = saved.get(file_key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            values[key] = float(value)
    return values


def save_model_settings(directory, settings):
    path = os.path.join(directory, MODEL_SETTINGS_FILENAME)
    write_json_atomic(
        path,
        {
            file_key: float(getattr(settings, key))
            for key, file_key in MODEL_SETTING_FILE_KEYS.items()
        },
    )


def write_json_atomic(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary_path = path + ".tmp"
    try:
        with open(temporary_path, "w", encoding="utf-8") as output:
            json.dump(data, output, ensure_ascii=False, indent=2)
        os.replace(temporary_path, path)
    except OSError:
        try:
            os.unlink(temporary_path)
        except OSError:
            pass
        raise
