"""Copy dropped voice descriptions (.causalvc.json) into the models folder.

Each becomes its own folder named after the file.  The description only points
at the voice's weights (see infer/causalvc_backend.py), so nothing large is copied.
"""

import os
import shutil
import uuid
from pathlib import Path

from tools.model_registry import VOICE_SUFFIXES


class ModelImportError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


def voice_name(path):
    """The voice's name: the file name without its .causalvc.json ending."""
    name = Path(path).name
    for suffix in VOICE_SUFFIXES:
        if name.endswith(suffix):
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
        raise ModelImportError(
            "import_no_model_file",
            "Drop a voice's .causalvc.json file (written when a voice is built).",
        )
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
