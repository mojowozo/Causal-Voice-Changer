"""Add dropped voices to the models folder.

A .cvc voice file (see causalvc/voice.py) is copied in as it is, into its
voice's folder: Kira-3000s.cvc goes to models/Kira/.  An older voice
description (.causalvc.json) is packed into a .cvc there.  Each voice is
named after its file.
"""

import os
import shutil
import sys
import uuid
from pathlib import Path

from tools.model_registry import PROJECT_ROOT, VOICE_SUFFIXES, discover_models, remove_if_empty, voice_folder_name

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
    """The voice files among the dropped paths."""
    files = [Path(path) for path in paths]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise ModelImportError("import_failed", f"File not found: {missing[0]}")
    voices = [path for path in files if voice_name(path) is not None]
    if not voices:
        raise ModelImportError("import_no_model_file", NO_VOICE_MESSAGE)
    return voices


def unique_name(models_root, name):
    """``name``, or ``name (2)`` and so on, so it clashes with no voice there (in any of its
    folders) and no file or folder at its top."""
    root = Path(models_root)
    taken = {entry.name.casefold() for entry in discover_models(root)}
    if root.is_dir():
        taken |= {path.name.casefold() for path in root.iterdir()}
    candidate = name
    number = 2
    while candidate.casefold() in taken or f"{candidate}{CVC_SUFFIX}".casefold() in taken:
        candidate = f"{name} ({number})"
        number += 1
    return candidate


def import_models(paths, models_root):
    """Add the dropped voices and return their names.

    Each is written under a hidden name and renamed into place only when
    complete, so an interrupted copy never shows up as a half-added voice.
    """
    voices = plan_import(paths)
    os.makedirs(models_root, exist_ok=True)
    names = []
    for voice in voices:
        base = voice_name(voice).strip() or "Voice"
        name = unique_name(models_root, base)
        folder = voice_files.voice_folder(models_root, voice_folder_name(base))
        target = folder / f"{name}{CVC_SUFFIX}"
        staging = folder / f".import-{uuid.uuid4().hex}{CVC_SUFFIX}"
        try:
            folder.mkdir(parents=True, exist_ok=True)
            if voice_files.is_cvc(voice):
                voice_files.read_cvc(voice)  # refuse a file that is not a voice
                shutil.copyfile(voice, staging)
            else:
                voice_files.folder_to_cvc(voice, staging, settings=True, name=name)
            os.replace(staging, target)
        except (OSError, ValueError, KeyError) as error:  # VoiceFileError is a ValueError
            staging.unlink(missing_ok=True)
            remove_if_empty(folder, models_root)
            raise ModelImportError("import_failed", str(error)) from error
        names.append(name)
    return names
