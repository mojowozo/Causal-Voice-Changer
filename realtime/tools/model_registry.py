import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

#: The repository the app lives in: relative paths in a voice description are resolved against it.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


#: Voice descriptions; the second spelling is the project's earlier name and still loads.
VOICE_SUFFIXES = (".causalvc.json", ".nextvc.json")


@dataclass(frozen=True)
class ModelEntry:
    name: str
    directory: Path
    model_path: Path
    #: Size of the voice's decoder file in bytes and the description's modification time (Unix).
    size_bytes: int = 0
    modified: float = 0.0


def discover_models(models_root: str | Path) -> list[ModelEntry]:
    """One voice per direct child directory that holds a voice description.

    When a folder has several, names are sorted case-insensitively and the
    first is used.
    """
    root = Path(models_root)
    if not root.is_dir():
        return []

    entries = []
    for directory in sorted(
        # Hidden folders are skipped, including in-progress imports (.import-*).
        (path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")),
        key=lambda path: path.name.casefold(),
    ):
        files = sorted(
            (path for path in directory.iterdir() if path.name.endswith(VOICE_SUFFIXES)),
            key=lambda path: path.name.casefold(),
        )
        if not files:
            continue
        try:
            modified = files[0].stat().st_mtime
        except OSError:
            continue
        entries.append(
            ModelEntry(
                name=directory.name,
                directory=directory.resolve(),
                model_path=files[0].resolve(),
                size_bytes=decoder_size(files[0]),
                modified=modified,
            )
        )
    return entries


def decoder_size(description: Path) -> int:
    """Size of the decoder file a voice description points at, or 0 if it cannot be read."""
    import json

    try:
        spec = json.loads(description.read_text(encoding="utf-8"))
        decoder = Path(spec["decoder"])
        if decoder.is_absolute():
            return decoder.stat().st_size
        for base in (description.parent, Path(spec.get("project") or PROJECT_ROOT)):  # beside the description first
            if (base / decoder).is_file():
                return (base / decoder).stat().st_size
        return 0
    except (OSError, ValueError, KeyError, TypeError):
        return 0


#: Characters a model folder name cannot contain (also invalid on NTFS/FAT).
INVALID_NAME_CHARACTERS = '/\\:*?"<>|'


def model_name_problem(name):
    """Why ``name`` cannot be a model folder name, or ``None`` if it can."""
    if not name or not name.strip():
        return "empty"
    if name != name.strip() or name.endswith("."):
        return "edges"
    if name.startswith("."):
        return "hidden"
    if any(character in INVALID_NAME_CHARACTERS or ord(character) < 32 for character in name):
        return "characters"
    if len(name.encode("utf-8")) > 200:
        return "too_long"
    return None


def rename_model_folder(directory, new_name):
    """Rename a model folder within its parent; returns the new path.

    Case-only renames (``voice`` -> ``Voice``) go through a temporary name,
    which case-insensitive filesystems such as NTFS need.
    """
    directory = Path(directory)
    target = directory.parent / new_name
    same_folder = target.exists() and os.path.samefile(target, directory)
    if target.exists() and not same_folder:
        raise FileExistsError(new_name)
    if same_folder and directory.name != new_name:
        temporary = directory.parent / f".rename-{os.getpid()}-{directory.name}"
        os.rename(directory, temporary)
        directory = temporary
    os.rename(directory, target)
    return target


class TrashError(Exception):
    pass


def move_to_trash(path):
    """Move a model folder to the desktop Trash (recoverable), via ``gio``."""
    gio = shutil.which("gio")
    if gio is None:
        raise TrashError("gio was not found, so the folder cannot be moved to the Trash")
    completed = subprocess.run(
        [gio, "trash", "--", os.fspath(path)], capture_output=True, text=True, timeout=60
    )
    if completed.returncode != 0:
        raise TrashError(completed.stderr.strip() or f"gio trash failed ({completed.returncode})")
