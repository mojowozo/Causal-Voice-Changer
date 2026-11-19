"""Convert CausalVC voices (folders) into single .cvc files you can share.

    python convert_to_cvc.py PATH [PATH ...] [--out FOLDER] [--no-settings]

Each PATH can be:
  - a voice folder (the one holding NAME.causalvc.json and decoder.pt),
  - a NAME.causalvc.json file itself, or
  - any folder holding voices, such as realtime/models or runs/models:
    every voice inside it is converted.

Each voice becomes NAME.cvc in the output folder (default: ./cvc-voices).
The original folders are never changed. A voice's saved realtime settings
(pitch, gains) go along unless you pass --no-settings.

A voice built on its own content encoder or pitch tracker (rather than the
shared ones every voice uses) carries that file too, so it sounds the same
on the other computer.

The decoder in a .cvc only holds the voice it is for. Older voice folders can
point at a full training checkpoint, which contains every speaker of the
base the voice was trained on; those are trimmed down to the one speaker, so
no other voice ends up in the file.

Needs Python 3.10+ and PyTorch (the realtime app's own Python has both).
Open a .cvc by dropping it on the realtime app's window.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import zipfile
from pathlib import Path

# The .cvc format (causalvc/voice.py): a zip with voice.json, decoder.pt and,
# optionally, realtime_settings.json.
CVC_SUFFIX = ".cvc"
CVC_FORMAT = "causalvc-voice"
CVC_VERSION = 1
DESCRIPTION_SUFFIXES = (".causalvc.json", ".nextvc.json")
SETTINGS = "realtime_settings.json"
#: A voice's own encoder or pitch tracker, carried only when it isn't the shared one.
PARTS = {"encoder": "encoder.pt", "pitch": "pitch.pt"}
#: Where a project keeps the shared ones: the downloaded base, or where they were trained.
SHARED_PLACES = {
    "encoder.pt": ("weights/causalvc-base/encoder.pt", "runs/models/enc-100hz-v4/encoder.pt"),
    "pitch.pt": ("weights/causalvc-base/pitch.pt", "runs/models/pitch-la1-v4/pitch.pt"),
}


class Skip(Exception):
    """A voice that can't be converted; the reason is shown and the rest carry on."""


def voice_name(description: Path) -> str:
    name = description.name
    for suffix in DESCRIPTION_SUFFIXES:
        name = name.removesuffix(suffix)
    return name


def descriptions(path: Path) -> list[Path]:
    """The voice descriptions a PATH stands for."""
    if path.is_file():
        if not path.name.endswith(DESCRIPTION_SUFFIXES):
            raise Skip(f"{path} is not a voice description (*.causalvc.json)")
        return [path]
    if not path.is_dir():
        raise Skip(f"{path} does not exist")
    here = sorted((p for p in path.iterdir() if p.name.endswith(DESCRIPTION_SUFFIXES)), key=lambda p: p.name.casefold())
    if here:
        return here[:1]  # a voice folder: one voice
    found = sorted(
        (p for suffix in DESCRIPTION_SUFFIXES for p in path.rglob(f"*{suffix}")),
        key=lambda p: str(p).casefold(),
    )
    seen, voices = set(), []
    for description in found:  # one voice per folder, as the app reads them
        if description.parent not in seen:
            seen.add(description.parent)
            voices.append(description)
    if not voices:
        raise Skip(f"no voices found in {path}")
    return voices


def find_decoder(description: Path, spec: dict) -> Path:
    value = spec.get("decoder")
    if not value:
        raise Skip(f"{description} names no decoder")
    recorded = Path(value)
    candidates = [recorded] if recorded.is_absolute() else [description.parent / recorded]
    if spec.get("project") and not recorded.is_absolute():
        candidates.append(Path(spec["project"]) / recorded)
    candidates.append(description.parent / recorded.name)  # written on another machine
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise Skip(f"{description.name} points at {value}, which is not on this computer")


def load_checkpoint(decoder: Path) -> dict:
    import torch

    try:
        return torch.load(decoder, map_location="cpu")
    except Exception as error:
        raise Skip(f"could not read {decoder}: {error}") from error


def decoder_for(decoder: Path, checkpoint: dict, speaker: str) -> bytes:
    """The decoder's bytes, trimmed to `speaker` if it holds other voices too."""
    import torch

    speakers = checkpoint.get("speakers") or []
    if speaker not in speakers:
        raise Skip(f"{decoder} has no speaker named {speaker!r}")
    if len(speakers) == 1:
        return decoder.read_bytes()  # already just this voice: copied as is
    index = speakers.index(speaker)
    generator = dict(checkpoint["generator"])
    generator["speaker.weight"] = generator["speaker.weight"][index : index + 1].clone()
    slim = {
        "generator": generator,
        "cfg": checkpoint["cfg"],
        "speakers": [speaker],
        "step": checkpoint.get("step", 0),
        "encoder": "encoder.pt",
    }
    buffer = io.BytesIO()
    torch.save(slim, buffer)
    print(f"  trimmed the decoder from {len(speakers)} speakers to just {speaker!r}")
    return buffer.getvalue()


def project_folders(description: Path, spec: dict) -> list[Path]:
    """Folders the voice's relative paths may start from: its own, the project it names, and every folder above it."""
    folders = [description.parent.resolve()]
    if spec.get("project"):
        folders.append(Path(spec["project"]))
    folders += list(description.parent.resolve().parents)
    return folders


def first_file(value, folders: list[Path]) -> Path | None:
    if not value:
        return None
    path = Path(value)
    for candidate in [path] if path.is_absolute() else [folder / path for folder in folders]:
        if candidate.is_file():
            return candidate
    return None


def digest(path: Path) -> bytes:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.digest()


def own_parts(description: Path, spec: dict, checkpoint: dict) -> dict[str, Path]:
    """The encoder and pitch tracker this voice uses that are not the shared ones."""
    folders = project_folders(description, spec)
    found = {
        "encoder": first_file(spec.get("encoder"), folders) or first_file(checkpoint.get("encoder"), folders[1:]),
        "pitch": first_file(spec.get("pitch"), folders),
    }
    parts = {}
    for key, path in found.items():
        if path is None:
            continue
        member = PARTS[key]
        shared = {p for folder in folders for place in SHARED_PLACES[member] if (p := folder / place).is_file()}
        if any(path.resolve() == copy.resolve() for copy in shared):
            continue
        if not any(path.stat().st_size == copy.stat().st_size and digest(path) == digest(copy) for copy in shared):
            parts[key] = path
            print(f"  includes the voice's own {key} ({path.stat().st_size // 1_000_000} MB)")
    return parts


def free_target(out: Path, name: str, taken: set[Path]) -> Path:
    target = out / f"{name}{CVC_SUFFIX}"
    number = 2
    while target in taken or target.exists():
        target = out / f"{name} ({number}){CVC_SUFFIX}"
        number += 1
    taken.add(target)
    return target


def write_cvc(target: Path, name: str, spec: dict, decoder: bytes, settings: bytes | None, parts: dict[str, Path]) -> None:
    meta = {"format": CVC_FORMAT, "version": CVC_VERSION, "name": name, "decoder": "decoder.pt", "speaker": spec["speaker"]}
    if spec.get("runtime"):
        meta["runtime"] = spec["runtime"]
    for key in parts:
        meta[key] = PARTS[key]
    partial = target.with_name(target.name + ".part")
    with zipfile.ZipFile(partial, "w") as archive:
        archive.writestr("voice.json", json.dumps(meta, indent=1), zipfile.ZIP_DEFLATED)
        archive.writestr("decoder.pt", decoder, zipfile.ZIP_STORED)  # already a zip of weights
        if settings is not None:
            archive.writestr(SETTINGS, settings, zipfile.ZIP_DEFLATED)
        for key, path in parts.items():
            archive.write(path, PARTS[key], zipfile.ZIP_STORED)
    os.replace(partial, target)


def convert(description: Path, out: Path, with_settings: bool, taken: set[Path]) -> Path:
    try:
        spec = json.loads(description.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise Skip(f"could not read {description}: {error}") from error
    if not isinstance(spec, dict) or not isinstance(spec.get("speaker"), str):
        raise Skip(f"{description} is not a CausalVC voice description")
    decoder_file = find_decoder(description, spec)
    checkpoint = load_checkpoint(decoder_file)
    decoder = decoder_for(decoder_file, checkpoint, spec["speaker"])
    parts = own_parts(description, spec, checkpoint)
    settings_file = description.parent / SETTINGS
    settings = settings_file.read_bytes() if with_settings and settings_file.is_file() else None
    name = voice_name(description)
    target = free_target(out, name, taken)
    write_cvc(target, name, spec, decoder, settings, parts)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert CausalVC voice folders into .cvc files.")
    parser.add_argument("paths", nargs="+", type=Path, help="voice folders, *.causalvc.json files, or folders of voices")
    parser.add_argument("--out", type=Path, default=Path("cvc-voices"), help="where to write the .cvc files (default: ./cvc-voices)")
    parser.add_argument("--no-settings", action="store_true", help="leave out each voice's saved realtime settings")
    args = parser.parse_args()

    try:
        import torch  # noqa: F401
    except ImportError:
        print("PyTorch is needed. Run this with the realtime app's Python, e.g. .venv/bin/python on Linux or .venv\\Scripts\\python.exe on Windows.")
        return 2

    voices, problems = [], []
    for path in args.paths:
        try:
            voices += descriptions(path.expanduser())
        except Skip as problem:
            problems.append(str(problem))
    args.out.mkdir(parents=True, exist_ok=True)
    taken: set[Path] = set()
    made = 0
    for description in voices:
        print(f"{voice_name(description)}  ({description.parent})")
        try:
            target = convert(description, args.out, not args.no_settings, taken)
        except Skip as problem:
            problems.append(str(problem))
            print(f"  skipped: {problem}")
            continue
        made += 1
        print(f"  -> {target}")
    print(f"\n{made} voice(s) written to {args.out.resolve()}")
    for problem in problems:
        print(f"skipped: {problem}")
    return 0 if made and not problems else 1


if __name__ == "__main__":
    sys.exit(main())
