"""Install dropped voices into the models folder.

A .cvc voice file (see causalvc/voice.py) is unpacked into a folder of its own.
An older voice description (.causalvc.json) is copied into one; it only points
at the voice's weights, so nothing large is copied.  Each folder is named after
the file.
"""

import os
import shutil
import sys
import uuid
from pathlib import Path

from tools.model_registry import PROJECT_ROOT, VOICE_SUFFIXES

if str(PROJECT_ROOT) not in sys.path:  # the causalvc package, when it is not installed in this environment
    sys.path.insert(0, str(PROJECT_ROOT))

from causalvc import voice as voice_files  # noqa: E402

CVC_SUFFIX = voice_files.CVC_SUFFIX
NO_VOICE_MESSAGE = "Drop a voice's .cvc file (written when a voice is built)."


class ModelImportError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


def voice_name(path):
    """The voice's name: the file name without its .cvc or .causalvc.json ending, or None
    for a file that is not a voice."""
    name = Path(path).name
    for suffix in (CVC_SUFFIX, *VOICE_SUFFIXES):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return None


def plan_import(paths):
    """The voice descriptions among the dropped paths."""
    files = [Path(path) for path in paths]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise ModelImportError("import_failed", f"File not found: {missing[0]}")
    voices = [path for path in files if voice_name(path) is not None]
    if not voices:
        raise ModelImportError("import_no_model_file", NO_VOICE_MESSAGE)
    return voices


def unique_folder_name(models_root, name):
    candidate = name
    number = 2
    while (Path(models_root) / candidate).exists():
        candidate = f"{name} ({number})"
        number += 1
    return candidate


def import_models(paths, models_root):
    """Copy the dropped voices in and return the new voice (folder) names.

    Files are copied into a hidden temporary folder that is renamed into
    place only when complete, so an interrupted copy never shows up as a
    half-imported voice.
    """
    voices = plan_import(paths)
    os.makedirs(models_root, exist_ok=True)
    names = []
    for voice in voices:
        name = unique_folder_name(models_root, voice_name(voice).strip() or "Voice")
        if voice_files.is_cvc(voice):
            try:
                voice_files.unpack(voice, Path(models_root) / name, name)
            except (OSError, ValueError) as error:  # VoiceFileError is a ValueError
                raise ModelImportError("import_failed", str(error)) from error
            names.append(name)
            continue
        # os.mkdir honours the umask (tempfile.mkdtemp would force 0700).
        staging = os.path.join(models_root, f".import-{uuid.uuid4().hex}")
        try:
            os.mkdir(staging)
            shutil.copy2(voice, os.path.join(staging, voice.name))
            os.rename(staging, os.path.join(models_root, name))
        except OSError as error:
            shutil.rmtree(staging, ignore_errors=True)
            raise ModelImportError("import_failed", str(error)) from error
        names.append(name)
    return names


def install_loose(models_root):
    """Install .cvc files put straight into the models folder; returns (names, problems).

    Each installed file is removed (its folder replaces it), so it is not
    installed again on the next reload.  One that is not a voice is left as is
    and reported."""
    root = Path(models_root)
    if not root.is_dir():
        return [], []
    names, problems = [], []
    for path in sorted(root.iterdir(), key=lambda path: path.name.casefold()):
        if not path.is_file() or path.name.startswith(".") or not voice_files.is_cvc(path):
            continue
        try:
            names += import_models([str(path)], root)
        except ModelImportError as error:
            problems.append(error.message)
            continue
        path.unlink(missing_ok=True)
    return names, problems
