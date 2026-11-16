"""Merge voices: a weighted average of their decoders' weights.

    PYTHONPATH=$PWD .venv/bin/python -m causalvc.merge OUT.cvc A.cvc=60 B.cvc=40

Every voice built in the training window is fine-tuned from the same base decoder, so
their weights stay close enough that an average between them is a voice too: the
result moves smoothly from one to the other as the shares change (as with RVC model
merging). The voice is in all of the decoder's weights, not only its speaker vector,
so every weight is averaged.

Voices can be merged when their decoders have the same layout (the same settings and
the same weights) and they use the same content encoder and pitch tracker: the shared
ones, or an identical file of their own. ``family`` says which voices go together,
reading only a few kilobytes of each file.
"""

from __future__ import annotations

import hashlib
import io
import json
import pickle
import sys
import zipfile
from collections import OrderedDict
from pathlib import Path

from causalvc import voice as voice_files


class MergeError(ValueError):
    """Voices that cannot be merged, with the reason."""


class _Stub:
    """Stands in for every object the metadata reader refuses to build."""

    def __init__(self, *args, **kwargs):
        pass


class _MetadataUnpickler(pickle.Unpickler):
    """Reads a torch checkpoint's structure without building anything: tensors become stubs
    and their bytes are never read, but plain values (the settings) and key names survive.
    (After RVC-model-merger's probe_checkpoint.)"""

    def find_class(self, module, name):
        return OrderedDict if (module, name) == ("collections", "OrderedDict") else _Stub

    def persistent_load(self, pid):
        return None


def _metadata(archive: zipfile.ZipFile) -> dict:
    """The decoder checkpoint's object graph minus its tensors."""
    with archive.open(voice_files.CVC_DECODER) as member, zipfile.ZipFile(member) as decoder:
        pkl = next(name for name in decoder.namelist() if name.endswith("data.pkl"))
        data = _MetadataUnpickler(io.BytesIO(decoder.read(pkl))).load()
    if not isinstance(data, dict) or not isinstance(data.get("generator"), dict):
        raise MergeError("not a decoder checkpoint")
    return data


def family(path) -> str | None:
    """A key that is equal for voices that can be merged with each other, or None for a file
    that cannot be merged (not a .cvc, or unreadable)."""
    path = Path(path)
    if not voice_files.is_cvc(path):
        return None
    try:
        with zipfile.ZipFile(path) as archive:
            data = _metadata(archive)
            # A voice's own encoder or pitch tracker must be the same file in every voice merged:
            # the zip's checksum and size of that member identify it without reading it.
            parts = {}
            for key, member in voice_files.CVC_PARTS.items():
                try:
                    info = archive.getinfo(member)
                    parts[key] = f"{info.CRC:08x}-{info.file_size}"
                except KeyError:
                    parts[key] = "shared"
    except (OSError, zipfile.BadZipFile, KeyError, StopIteration, pickle.UnpicklingError, EOFError,
            AttributeError, MergeError, ValueError):
        return None
    layout = {"cfg": data.get("cfg"), "weights": sorted(data["generator"]), "parts": parts}
    return hashlib.sha1(json.dumps(layout, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _decoder(path: Path) -> tuple[dict, dict]:
    import torch

    meta = voice_files.read_cvc(path)
    with zipfile.ZipFile(path) as archive:
        checkpoint = torch.load(io.BytesIO(archive.read(voice_files.CVC_DECODER)), map_location="cpu")
    return meta, checkpoint


def merge(voices: list[tuple[Path, float]], target, name: str, settings: bytes | None = None) -> Path:
    """Write the weighted average of ``voices`` ((path, strength) pairs; the strengths are
    shares, so 60 and 40 is the same as 3 and 2) to ``target`` as a .cvc named ``name``.
    The first voice's own encoder or pitch tracker, if any, goes along (all must share it)."""
    import torch

    voices = [(Path(path), float(strength)) for path, strength in voices if float(strength) > 0]
    if len(voices) < 2:
        raise MergeError("Choose at least two voices with a strength above zero.")
    families = {family(path) for path, _ in voices}
    if None in families:
        raise MergeError("Only .cvc voices can be merged.")
    if len(families) > 1:
        raise MergeError("These voices cannot be merged: they were built differently (another base or settings).")
    total = sum(strength for _, strength in voices)
    first_meta, first = _decoder(voices[0][0])
    sums = {key: torch.zeros_like(value, dtype=torch.float32) for key, value in first["generator"].items()
            if value.is_floating_point()}
    sources = []
    for path, strength in voices:
        meta, checkpoint = _decoder(path)
        share = strength / total
        sources.append({"name": meta["name"], "share": round(share, 4)})
        for key, accumulated in sums.items():
            value = checkpoint["generator"][key]
            if value.shape != accumulated.shape:
                raise MergeError(f"{meta['name']} has a different shape for {key}.")
            accumulated += value.float() * share
    generator = {key: sums[key].to(value.dtype) if key in sums else value
                 for key, value in first["generator"].items()}
    merged = {**first, "generator": generator, "speakers": [name], "step": 0, "merged": sources}
    buffer = io.BytesIO()
    torch.save(merged, buffer)
    parts = {}
    with zipfile.ZipFile(voices[0][0]) as archive:
        staged = Path(target).with_name(f".{Path(target).name}.parts")
        for key, member in voice_files.CVC_PARTS.items():
            if member in archive.namelist():
                staged.mkdir(parents=True, exist_ok=True)
                parts[key] = staged / member
                parts[key].write_bytes(archive.read(member))
    try:
        return voice_files.write_cvc(target, name, name, buffer.getvalue(), settings, first_meta.get("runtime"), parts)
    finally:
        for path in parts.values():
            path.unlink(missing_ok=True)
        if parts:
            next(iter(parts.values())).parent.rmdir()


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    if len(args) < 3:
        sys.exit(__doc__.split("\n\n")[1])
    target, voices = Path(args[0]), []
    for item in args[1:]:
        path, _, strength = item.rpartition("=")
        voices.append((Path(path or item), float(strength) if path else 1.0))
    written = merge(voices, target, voice_files.voice_name(target))
    print(f"wrote {written}")


if __name__ == "__main__":
    main()
