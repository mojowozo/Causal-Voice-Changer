"""Engine settings, defaults, and their JSON persistence.

Global settings live in ``configs/engine.json``.  The per-voice settings
(gains, noise gate, pitch and the advanced pitch settings) live in ``configs/voice_settings.json``, by voice
name; a voice with none there starts from the ones it carries (the
``realtime_settings.json`` inside its .cvc, or in an older voice folder).

Every checkpoint of a voice (``models/Kira/Kira-1000s.cvc`` ...) shares the voice's entry,
which also remembers the checkpoint picked last when it is not the newest
(``VOICE_CHECKPOINT_KEY``).  Entries saved by checkpoint name before checkpoints were
grouped are left in the file; ``migrate_voice_settings`` copies one into the voice's entry.
"""

import json
import math
import os
from dataclasses import asdict, dataclass, fields

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "engine.json")
VOICE_SETTINGS_PATH = os.path.join(PROJECT_ROOT, "configs", "voice_settings.json")
DEFAULT_RECORDING_FOLDER = os.path.join(PROJECT_ROOT, "recordings")

INPUT_SOURCES = ("microphone", "file")
RECORDING_MODES = ("separate", "mix", "stereo")
#: Output-level indicator targets (tools/loudness.py TARGETS_LUFS).
LOUDNESS_TARGETS = ("voice_chat", "streaming")
#: Input noise reduction: Hushwire (Amazon Voice Focus, ~53 ms) or RNNoise (20 ms).
INPUT_DENOISE_METHODS = ("hushwire", "rnnoise")
#: The largest pitch offset, in Hz, either way, and the highest pitch ceiling (0 = no ceiling).
PITCH_OFFSET_LIMIT = 300.0
PITCH_CEILING_LIMIT = 1000.0
#: The advanced pitch settings, applied together (CausalVC.change_pitch_map).
PITCH_MAP_KEYS = ("pitch_offset_hz", "pitch_ceiling_hz", "pitch_ceiling_keep")
#: ASIO buffer sizes (samples) the app can ask the driver for; 0 keeps the driver's own setting.
ASIO_BUFFER_SIZES = (0, 32, 64, 96, 128, 192, 256, 512, 1024, 2048)

# Protocol name -> key in a voice's saved settings (voice_settings.json, realtime_settings.json).
MODEL_SETTING_FILE_KEYS = {
    "input_gain_db": "input_gain_db",
    "output_gain_db": "output_gain_db",
    "monitor_gain_db": "monitor_gain_db",
    "noise_gate_db": "threhold",
    "pitch": "pitch",
    "pitch_offset_hz": "pitch_offset_hz",
    "pitch_ceiling_hz": "pitch_ceiling_hz",
    "pitch_ceiling_keep": "pitch_ceiling_keep",
    "index_rate": "index_rate",
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
        "cuda_graphs",
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
    #: Record each block's GPU work once and replay it (a CUDA graph): about half the time per block
    #: and far steadier. Falls back on its own where recording fails; off runs block by block.
    cuda_graphs: bool = True
    #: Stretch sudden treble in the middle of a word (a voice's 4-12 kHz band switching on within
    #: 2 ms, heard as a click) over a few milliseconds; adds 5 ms of delay. See tools/treble_softener.py.
    soften_treble: bool = True
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
    #: Hz added to the shifted pitch of voiced frames: output pitch = input * 2^(pitch/12) + this.
    pitch_offset_hz: float = 0.0
    #: Output pitch above which only ``pitch_ceiling_keep`` (0-1) of the rise is kept, in semitones;
    #: 0 is no ceiling. Pulls down peaks while notes below it keep their intervals.
    pitch_ceiling_hz: float = 0.0
    pitch_ceiling_keep: float = 0.25
    #: How much of what is said is drawn from the voice's own audio (its index), 0-1; 0 is off.
    index_rate: float = 0.0

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
    "cuda_graphs",
    "soften_treble",
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
        if key == "pitch_offset_hz":
            value = min(max(value, -PITCH_OFFSET_LIMIT), PITCH_OFFSET_LIMIT)
        if key == "pitch_ceiling_hz":
            value = min(max(value, 0.0), PITCH_CEILING_LIMIT)
        if key in ("pitch_ceiling_keep", "index_rate"):
            value = min(max(value, 0.0), 1.0)
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


def model_settings_from(saved):
    """One voice's settings (protocol keys) from what was saved for it (file keys), over the defaults."""
    defaults = EngineSettings()
    values = {key: getattr(defaults, key) for key in MODEL_SETTING_KEYS}
    if not isinstance(saved, dict):
        return values
    for key, file_key in MODEL_SETTING_FILE_KEYS.items():
        value = saved.get(file_key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            values[key] = float(value)
    return values


def model_settings_to_save(values):
    """One voice's settings (protocol keys, or an EngineSettings) in their saved form (file keys)."""
    get = values.get if isinstance(values, dict) else lambda key: getattr(values, key)
    return {file_key: float(get(key)) for key, file_key in MODEL_SETTING_FILE_KEYS.items()}


def load_voice_settings(path=VOICE_SETTINGS_PATH):
    """Every voice's saved settings: {voice name: {file key: value}}."""
    try:
        with open(path, "r", encoding="utf-8") as settings_file:
            saved = json.load(settings_file)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        print(f"Could not read voice settings: {path}")
        return {}
    if not isinstance(saved, dict):
        return {}
    return {name: values for name, values in saved.items() if isinstance(name, str) and isinstance(values, dict)}


#: In a voice's entry: the checkpoint the user picked last, when it is not the newest one.
VOICE_CHECKPOINT_KEY = "checkpoint"


def has_model_settings(saved):
    """Whether a saved entry holds any per-voice setting (it may hold only the picked checkpoint)."""
    return isinstance(saved, dict) and any(file_key in saved for file_key in MODEL_SETTING_FILE_KEYS.values())


def migrate_voice_settings(store, groups, selected=""):
    """Give every voice with checkpoints (``tools.model_registry.VoiceGroup``) the settings that were
    saved per checkpoint, when it has none of its own: the selected checkpoint's (``selected``, the one
    in use last), else the newest checkpoint's that has any; a selected checkpoint that is not the
    newest also becomes the voice's picked one.  The checkpoint entries are kept.  Returns whether
    ``store`` changed."""
    changed = False
    for group in groups:
        names = [entry.name for entry in group.checkpoints]
        voice = store.get(group.name)
        if not has_model_settings(voice):
            candidates = [name for name in reversed(names) if name != group.name and has_model_settings(store.get(name))]
            if selected in candidates:
                candidates.insert(0, selected)
            if candidates:
                entry = dict(voice or {})
                entry.update({key: value for key, value in store[candidates[0]].items()
                              if key in MODEL_SETTING_FILE_KEYS.values()})
                if len(names) > 1 and selected in names and selected != group.latest.name:
                    entry.setdefault(VOICE_CHECKPOINT_KEY, selected)
                store[group.name] = entry
                changed = True
    return changed


def save_voice_settings(store, path=VOICE_SETTINGS_PATH):
    write_json_atomic(path, dict(sorted(store.items(), key=lambda item: item[0].casefold())))


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
