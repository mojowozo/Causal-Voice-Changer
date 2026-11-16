import json
import re
import os
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path

#: The repository the app lives in: relative paths in a voice description are resolved against it.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


#: Voice descriptions; the second spelling is the project's earlier name and still loads.
VOICE_SUFFIXES = (".causalvc.json", ".nextvc.json")
#: A voice packed into one file (causalvc/voice.py), the way voices are kept now.
CVC_SUFFIX = ".cvc"
#: Per-voice settings inside a .cvc or an older voice folder.
SETTINGS_FILENAME = "realtime_settings.json"


@dataclass(frozen=True)
class ModelEntry:
    name: str
    #: What the voice is on disk: its .cvc file, or (older installs) its folder.
    path: Path
    #: What the runtime loads: the .cvc, or the folder's description file.
    model_path: Path
    #: Size of the voice (its .cvc, or the folder's decoder) in bytes, and its modification time (Unix).
    size_bytes: int = 0
    modified: float = 0.0

    @property
    def is_file(self):
        return self.path.suffix.lower() == CVC_SUFFIX and not self.path.is_dir()


def discover_models(models_root: str | Path) -> list[ModelEntry]:
    """The voices in the models folder, each named after its file:

        models/Kira/Kira-1000s.cvc    a voice's folder of checkpoints (how builds add them)
        models/Kira-1000s.cvc         a .cvc put straight in the folder
        models/Kira-1000s/            an older install: a folder holding a voice description

    A folder holding a voice description is an older voice, not a voice's
    folder.  When two voices share a name (ignoring case) the first found is
    used: loose files, then folders in name order.  When an older folder has
    several descriptions, the first by name is used.  Hidden entries are
    skipped (in-progress copies and imports).
    """
    root = Path(models_root)
    if not root.is_dir():
        return []

    entries = []
    seen = set()

    def visible(path):
        return not path.name.startswith(".")

    def add_file(path):
        name = path.name[: -len(CVC_SUFFIX)]
        if name.casefold() in seen:
            return
        try:
            stat = path.stat()
        except OSError:
            return
        seen.add(name.casefold())
        entries.append(
            ModelEntry(name=name, path=path.resolve(), model_path=path.resolve(),
                       size_bytes=stat.st_size, modified=stat.st_mtime)
        )

    def cvc_files(folder):
        return sorted(
            (path for path in folder.iterdir()
             if visible(path) and path.is_file() and path.name.lower().endswith(CVC_SUFFIX)),
            key=lambda path: path.name.casefold(),
        )

    for path in cvc_files(root):
        add_file(path)
    for directory in sorted(
        (path for path in root.iterdir() if path.is_dir() and visible(path)),
        key=lambda path: path.name.casefold(),
    ):
        descriptions = sorted(
            (path for path in directory.iterdir() if path.name.endswith(VOICE_SUFFIXES)),
            key=lambda path: path.name.casefold(),
        )
        if not descriptions:
            for path in cvc_files(directory):
                add_file(path)
            continue
        if directory.name.casefold() in seen:
            continue
        try:
            modified = descriptions[0].stat().st_mtime
        except OSError:
            continue
        seen.add(directory.name.casefold())
        entries.append(
            ModelEntry(
                name=directory.name,
                path=directory.resolve(),
                model_path=descriptions[0].resolve(),
                size_bytes=decoder_size(descriptions[0]),
                modified=modified,
            )
        )
    return sorted(entries, key=lambda entry: entry.name.casefold())


def voice_folder_name(name):
    """The voice a checkpoint's name belongs to: ``Kira`` for ``Kira-1000s``; a name
    without a step count is its own voice."""
    match = re.fullmatch(r"(.+?)-\d+s", name)
    return match.group(1) if match else name


def embedded_settings(model: ModelEntry):
    """The settings a voice came with (inside its .cvc, or its folder's file), as a dict, or None."""
    try:
        if model.is_file:
            with zipfile.ZipFile(model.path) as archive:
                data = archive.read(SETTINGS_FILENAME)
        else:
            data = (model.path / SETTINGS_FILENAME).read_bytes()
        saved = json.loads(data.decode("utf-8"))
    except (KeyError, OSError, ValueError, zipfile.BadZipFile):
        return None
    return saved if isinstance(saved, dict) else None


def decoder_size(description: Path) -> int:
    """Size of the decoder file a voice description points at, or 0 if it cannot be read."""
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


def rename_model(path, new_name):
    """Rename a voice (its .cvc, keeping the ending, or an older voice's folder) within
    its folder; returns the new path.

    Case-only renames (``voice`` -> ``Voice``) go through a temporary name,
    which case-insensitive filesystems such as NTFS need.
    """
    directory = Path(path)
    suffix = CVC_SUFFIX if directory.is_file() else ""
    target = directory.parent / f"{new_name}{suffix}"
    same_folder = target.exists() and os.path.samefile(target, directory)
    if target.exists() and not same_folder:
        raise FileExistsError(new_name)
    if same_folder and directory.name != new_name:
        temporary = directory.parent / f".rename-{os.getpid()}-{directory.name}"
        os.rename(directory, temporary)
        directory = temporary
    os.rename(directory, target)
    return target


def remove_if_empty(folder, models_root):
    """Remove a voice's folder once nothing is left in it (never the models folder itself)."""
    folder, root = Path(folder), Path(models_root)
    try:
        if folder.resolve() != root.resolve() and folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()
    except OSError:
        pass


class TrashError(Exception):
    pass


def move_to_recycle_bin(path):
    """Windows: move ``path`` to the Recycle Bin through the shell (recoverable, no prompt)."""
    import ctypes
    from ctypes import wintypes

    class SHFILEOPSTRUCTW(ctypes.Structure):
        _fields_ = [
            ("hwnd", wintypes.HWND),
            ("wFunc", wintypes.UINT),
            ("pFrom", wintypes.LPCWSTR),
            ("pTo", wintypes.LPCWSTR),
            ("fFlags", ctypes.c_uint16),
            ("fAnyOperationsAborted", wintypes.BOOL),
            ("hNameMappings", ctypes.c_void_p),
            ("lpszProgressTitle", wintypes.LPCWSTR),
        ]

    FO_DELETE = 3
    FOF_SILENT, FOF_NOCONFIRMATION, FOF_ALLOWUNDO, FOF_NOERRORUI = 0x4, 0x10, 0x40, 0x400
    operation = SHFILEOPSTRUCTW(
        wFunc=FO_DELETE,
        # A list of paths, each NUL-terminated, ending with an extra NUL.
        pFrom=os.path.abspath(os.fspath(path)) + "\0",
        fFlags=FOF_SILENT | FOF_NOCONFIRMATION | FOF_ALLOWUNDO | FOF_NOERRORUI,
    )
    result = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(operation))
    if result != 0 or operation.fAnyOperationsAborted:
        raise TrashError(f"the Recycle Bin refused it (error {result:#x})")


def move_to_trash(path):
    """Move a voice (its file or folder) to the desktop Trash (recoverable): ``gio`` on Linux,
    the Recycle Bin on Windows."""
    if os.name == "nt":
        move_to_recycle_bin(path)
        return
    gio = shutil.which("gio")
    if gio is None:
        raise TrashError("gio was not found, so the voice cannot be moved to the Trash")
    completed = subprocess.run(
        [gio, "trash", "--", os.fspath(path)], capture_output=True, text=True, timeout=60
    )
    if completed.returncode != 0:
        raise TrashError(completed.stderr.strip() or f"gio trash failed ({completed.returncode})")
