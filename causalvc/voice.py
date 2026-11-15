"""Voices as self-contained files and folders.

A voice is shared as one file, NAME-10000s.cvc: a zip holding

    voice.json                  {"format": "causalvc-voice", "version": 1, "name": "NAME-10000s",
                                 "decoder": "decoder.pt", "speaker": "NAME"}
    decoder.pt                  the voice's decoder, this one speaker only
    realtime_settings.json      optional: the pitch and gains it was used with
    encoder.pt, pitch.pt        optional: only for a voice built on its own content encoder or
                                pitch tracker instead of the shared ones (voice.json then
                                names them: "encoder": "encoder.pt")

and installed (by the realtime app, or unpack()) as a folder of the same files,
the description renamed to NAME-10000s.causalvc.json. Nothing in either names
a place on the machine that made it, so both work anywhere this project is
installed. The two networks every voice shares (the content encoder and the
pitch tracker) are not included unless the voice uses its own: they come with the
base that setup downloads.

    python -m causalvc.voice cvc PATH [--out FOLDER]      voice folders -> .cvc files
    python -m causalvc.voice unpack FILE.cvc --out FOLDER  a .cvc -> an installed folder
    python -m causalvc.voice pack PATH [--out FOLDER]     an older voice file (pointing at
                                                           weights elsewhere) -> a folder
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import uuid
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUFFIXES = (".causalvc.json", ".nextvc.json")
#: A voice packed into one file.
CVC_SUFFIX = ".cvc"
CVC_FORMAT = "causalvc-voice"
CVC_VERSION = 1
#: The only members a .cvc may hold; anything else is ignored, so a file can never write elsewhere.
CVC_DESCRIPTION = "voice.json"
CVC_DECODER = "decoder.pt"
CVC_SETTINGS = "realtime_settings.json"
#: A voice's own encoder or pitch tracker, carried only when it doesn't use the shared one.
CVC_PARTS = {"encoder": "encoder.pt", "pitch": "pitch.pt"}
#: Far above any real decoder (tens of MB); a bigger member is not a voice.
CVC_MAX_MEMBER_BYTES = 2_000_000_000


class VoiceFileError(ValueError):
    """A file that is not a usable .cvc voice."""


#: Where the shared networks were trained, on a machine that has them there instead of the downloaded base.
SHARED_LOCAL = {"encoder.pt": "runs/models/enc-la2-460/encoder.pt", "pitch.pt": "runs/models/pitch-la2/pitch.pt"}


def shared(name: str) -> Path:
    """The content encoder ("encoder.pt") or pitch tracker ("pitch.pt") every voice uses."""
    from causalvc.weights import base_folder, ensure_base

    for path in (base_folder() / name, ROOT / SHARED_LOCAL[name]):  # the downloaded base, or where they were trained
        if path.is_file():
            return path
    return ensure_base() / name


def existing(value, *bases: Path) -> Path | None:
    """`value` as a file, tried as given and under each base; None when it is nowhere."""
    if not value:
        return None
    path = Path(value)
    for candidate in [path] if path.is_absolute() else [base / path for base in bases]:
        if candidate.is_file():
            return candidate
    return None


def encoder_path(recorded=None) -> str:
    """The encoder a decoder was trained with if that file is here, else the shared one.
    Checkpoints may record a path from the machine that trained them; that is only a hint."""
    return str(existing(recorded, ROOT) or shared("encoder.pt"))


def voice_name(path) -> str:
    name = Path(path).name
    for suffix in (*SUFFIXES, CVC_SUFFIX):
        name = name.removesuffix(suffix)
    return name


def is_cvc(path) -> bool:
    return str(path).lower().endswith(CVC_SUFFIX)


def load(description) -> dict:
    """A voice (.cvc file or description file) -> {"decoder", "pitch", "encoder", "speaker", "runtime"} with real paths.

    A .cvc is unpacked once into a cache folder. Relative paths are looked for
    beside the description first, then under the project folder the file names
    (older files), then under this one."""
    description = Path(description)
    if is_cvc(description):
        description = cached(description)
    spec, bases, decoder = read_description(description)
    return {
        "decoder": str(decoder),
        "pitch": str(existing(spec.get("pitch"), *bases) or shared("pitch.pt")),
        "encoder": str(existing(spec.get("encoder"), *bases)) if existing(spec.get("encoder"), *bases) else None,
        "speaker": spec["speaker"],
        "runtime": spec.get("runtime"),
    }


def slim_decoder(decoder, speaker: str) -> dict:
    """A decoder checkpoint cut down to `speaker`, so the voices of the base it was built on do not travel with it."""
    import torch

    ck = torch.load(decoder, map_location="cpu")
    index = ck["speakers"].index(speaker)
    generator = dict(ck["generator"])
    generator["speaker.weight"] = generator["speaker.weight"][index : index + 1].clone()
    return {"generator": generator, "cfg": ck["cfg"], "speakers": [speaker], "step": ck.get("step", 0), "encoder": "encoder.pt"}


def read_description(description: Path) -> tuple[dict, list[Path], Path]:
    """A description's contents, the folders its relative paths are tried under, and its decoder file."""
    spec = json.loads(description.read_text(encoding="utf-8"))
    bases = [description.parent] + ([Path(spec["project"])] if spec.get("project") else []) + [ROOT]
    decoder = existing(spec["decoder"], *bases)
    if decoder is None and Path(spec["decoder"]).is_absolute():  # written on another machine: try the file name here
        decoder = existing(Path(spec["decoder"]).name, description.parent)
    if decoder is None:
        raise FileNotFoundError(f"{description.name} points at {spec['decoder']}, which is not on this machine. "
                                "A voice folder should contain its own decoder.pt (python -m causalvc.voice pack).")
    return spec, bases, decoder


def export(decoder, speaker: str, folder, name: str | None = None) -> Path:
    """Write a self-contained voice folder from a decoder checkpoint; returns the description's path."""
    import torch

    folder = Path(folder)
    name = name or folder.name
    slim = slim_decoder(decoder, speaker)
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(slim, folder / "decoder.pt.part")
    (folder / "decoder.pt.part").replace(folder / "decoder.pt")
    for old in folder.iterdir():  # one description per voice folder
        if old.name.endswith(SUFFIXES):
            old.unlink()
    target = folder / f"{name}.causalvc.json"
    target.write_text(json.dumps({"decoder": "decoder.pt", "speaker": speaker}, indent=1), encoding="utf-8")
    return target


def write_cvc(target, name: str, speaker: str, decoder_bytes: bytes, settings: bytes | None = None,
              runtime: str | None = None, parts: dict | None = None) -> Path:
    """Write a .cvc file atomically (a half-written file never has the final name).

    `parts` maps "encoder"/"pitch" to a voice's own network file, for one that doesn't use the shared ones."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    meta = {"format": CVC_FORMAT, "version": CVC_VERSION, "name": name, "decoder": CVC_DECODER, "speaker": speaker}
    if runtime:
        meta["runtime"] = runtime
    parts = parts or {}
    for key in parts:
        meta[key] = CVC_PARTS[key]
    partial = target.with_name(target.name + ".part")
    with zipfile.ZipFile(partial, "w") as archive:
        archive.writestr(CVC_DESCRIPTION, json.dumps(meta, indent=1), zipfile.ZIP_DEFLATED)
        # The decoder is itself a zip of float weights: storing it as is costs little and loads fast.
        archive.writestr(CVC_DECODER, decoder_bytes, zipfile.ZIP_STORED)
        if settings is not None:
            archive.writestr(CVC_SETTINGS, settings, zipfile.ZIP_DEFLATED)
        for key, path in parts.items():
            archive.write(path, CVC_PARTS[key], zipfile.ZIP_STORED)
    partial.replace(target)
    return target


def export_cvc(decoder, speaker: str, target, name: str | None = None) -> Path:
    """Write a .cvc voice from a decoder checkpoint; returns its path."""
    import torch

    target = Path(target)
    buffer = io.BytesIO()
    torch.save(slim_decoder(decoder, speaker), buffer)
    return write_cvc(target, name or voice_name(target), speaker, buffer.getvalue())


def folder_to_cvc(path, target, settings: bool = True) -> Path:
    """Pack an installed voice (its folder, or the description in it) into a .cvc file.

    With `settings`, the folder's realtime_settings.json (pitch, gains) goes along."""
    path = Path(path)
    description = description_in(path) if path.is_dir() else path
    spec, bases, decoder = read_description(description)
    extra = description.parent / CVC_SETTINGS
    return write_cvc(
        target,
        voice_name(target),
        spec["speaker"],
        own_decoder(decoder, spec["speaker"]),
        extra.read_bytes() if settings and extra.is_file() else None,
        spec.get("runtime"),
        own_parts(spec, bases, decoder),
    )


def own_decoder(decoder, speaker: str) -> bytes:
    """The decoder's bytes, holding only `speaker`.

    Older voice folders can point at a full training checkpoint with every speaker of
    its base; a shared file must never carry those, so such a decoder is trimmed."""
    import torch

    decoder = Path(decoder)
    speakers = torch.load(decoder, map_location="cpu").get("speakers") or []
    if speakers == [speaker]:
        return decoder.read_bytes()
    if speaker not in speakers:
        raise VoiceFileError(f"{decoder.name} has no speaker named {speaker!r}")
    buffer = io.BytesIO()
    torch.save(slim_decoder(decoder, speaker), buffer)
    return buffer.getvalue()


def _same_file(a: Path, b: Path) -> bool:
    if a.resolve() == b.resolve():
        return True
    if a.stat().st_size != b.stat().st_size:
        return False
    digests = []
    for path in (a, b):
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        digests.append(digest.digest())
    return digests[0] == digests[1]


def own_parts(spec: dict, bases: list[Path], decoder) -> dict:
    """The encoder and pitch tracker a voice uses here that are not the shared ones: {"encoder": path, ...}.

    Found the way load() and the runtimes find them: the description first, then (encoder) the
    path the checkpoint recorded. Compared with the shared files already on this machine, so
    nothing is downloaded."""
    import torch
    from causalvc.weights import base_folder

    recorded = torch.load(decoder, map_location="cpu").get("encoder")
    found = {
        "encoder": existing(spec.get("encoder"), *bases) or existing(recorded, ROOT),
        "pitch": existing(spec.get("pitch"), *bases),
    }
    parts = {}
    for key, path in found.items():
        if path is None:
            continue
        member = CVC_PARTS[key]
        shared_copies = [p for p in (base_folder() / member, ROOT / SHARED_LOCAL[member]) if p.is_file()]
        if not any(_same_file(path, copy) for copy in shared_copies):
            parts[key] = path
    return parts


def description_in(folder) -> Path:
    found = sorted((p for p in Path(folder).iterdir() if p.name.endswith(SUFFIXES)), key=lambda p: p.name.casefold())
    if not found:
        raise FileNotFoundError(f"no voice description in {folder}")
    return found[0]


def read_cvc(path) -> dict:
    """The description inside a .cvc file, checked: raises VoiceFileError if it is not a voice."""
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            if CVC_DESCRIPTION not in names or CVC_DECODER not in names:
                raise VoiceFileError(f"{Path(path).name} is not a voice file (no {CVC_DESCRIPTION} or {CVC_DECODER} inside)")
            meta = json.loads(archive.read(CVC_DESCRIPTION).decode("utf-8"))
    except (zipfile.BadZipFile, UnicodeDecodeError, json.JSONDecodeError, OSError) as error:
        raise VoiceFileError(f"{Path(path).name} is not a voice file ({error})") from error
    if not isinstance(meta, dict) or meta.get("format") != CVC_FORMAT or not isinstance(meta.get("speaker"), str):
        raise VoiceFileError(f"{Path(path).name} is not a CausalVC voice file")
    if int(meta.get("version", 0)) > CVC_VERSION:
        raise VoiceFileError(f"{Path(path).name} was made by a newer version of CausalVC; update this one to use it")
    return meta


def unpack(path, folder, name: str | None = None) -> Path:
    """Install a .cvc as a voice folder (created; must not exist yet); returns the description's path.

    Only the known members are extracted, each to a fixed name, so the archive
    cannot write anywhere else. The folder appears complete or not at all."""
    path, folder = Path(path), Path(folder)
    meta = read_cvc(path)
    name = name or folder.name
    staging = folder.with_name(f".unpack-{uuid.uuid4().hex}")
    staging.mkdir(parents=True)
    try:
        with zipfile.ZipFile(path) as archive:
            for member in (CVC_DECODER, CVC_SETTINGS, *CVC_PARTS.values()):
                try:
                    info = archive.getinfo(member)
                except KeyError:
                    continue
                if info.file_size > CVC_MAX_MEMBER_BYTES:
                    raise VoiceFileError(f"{path.name}: {member} is too large to be a voice")
                with archive.open(info) as source, open(staging / member, "wb") as sink:
                    shutil.copyfileobj(source, sink, 1 << 20)
        description = {"decoder": CVC_DECODER, "speaker": meta["speaker"]}
        for key, member in CVC_PARTS.items():
            if meta.get(key) == member and (staging / member).is_file():
                description[key] = member
        if meta.get("runtime"):
            description["runtime"] = meta["runtime"]
        (staging / f"{name}.causalvc.json").write_text(json.dumps(description, indent=1), encoding="utf-8")
        os.rename(staging, folder)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return folder / f"{name}.causalvc.json"


def install(path, folder) -> Path:
    """Install a .cvc as `folder`, replacing the voice files of one already there (a rebuild
    of the same checkpoint) but keeping its realtime settings. Returns the description's path."""
    folder = Path(folder)
    if not folder.exists():
        return unpack(path, folder)
    fresh = unpack(path, folder.with_name(f".install-{uuid.uuid4().hex}"), folder.name).parent
    try:
        for old in folder.iterdir():
            if old.name.endswith(SUFFIXES):
                old.unlink()
        for item in fresh.iterdir():
            if item.name == CVC_SETTINGS and (folder / CVC_SETTINGS).exists():
                continue  # the settings chosen on this machine win over the file's
            item.replace(folder / item.name)
    finally:
        shutil.rmtree(fresh, ignore_errors=True)
    return description_in(folder)


def cache_folder() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "causalvc" / "voices"


def cached(path) -> Path:
    """A .cvc unpacked into the cache (once per version of the file); returns the description's path."""
    path = Path(path).resolve()
    stat = path.stat()
    key = hashlib.sha1(f"{path}|{stat.st_size}|{stat.st_mtime_ns}".encode()).hexdigest()[:16]
    name = voice_name(path)
    folder = cache_folder() / f"{name}-{key}"
    description = folder / f"{name}.causalvc.json"
    if not description.is_file():
        folder.parent.mkdir(parents=True, exist_ok=True)
        try:
            unpack(path, folder, name)
        except FileExistsError:  # unpacked by another process meanwhile
            pass
    return description


def pack(path, out=None) -> Path:
    """Make an older voice (a description file, or the folder holding one) self-contained.
    In place by default; with `out`, as a new folder there and the original untouched."""
    path = Path(path)
    if path.is_dir():
        found = sorted(p for p in path.iterdir() if p.name.endswith(SUFFIXES))
        if not found:
            raise FileNotFoundError(f"no voice description in {path}")
        path = found[0]
    spec = load(path)
    name = path.parent.name if out is None else voice_name(path)
    folder = path.parent if out is None else Path(out) / name
    if out is not None and path.parent != folder:
        folder.mkdir(parents=True, exist_ok=True)
        settings = path.parent / "realtime_settings.json"
        if settings.is_file():
            shutil.copy2(settings, folder / settings.name)
    return export(spec["decoder"], spec["speaker"], folder, name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack", help="make an older voice self-contained")
    p.add_argument("path", nargs="+", help="voice description files, or folders holding one")
    p.add_argument("--out", help="write the packed voices into this folder instead of in place")
    c = sub.add_parser("cvc", help="pack voice folders into .cvc files, to share")
    c.add_argument("path", nargs="+", help="voice folders, or the description files in them")
    c.add_argument("--out", help="folder for the .cvc files (default: beside each voice folder)")
    c.add_argument("--no-settings", action="store_true", help="leave out the realtime settings (pitch, gains)")
    u = sub.add_parser("unpack", help="install a .cvc file as a voice folder")
    u.add_argument("path", nargs="+", help=".cvc files")
    u.add_argument("--out", required=True, help="folder to put the voice folders in (e.g. realtime/models)")
    a = ap.parse_args()
    for path in a.path:
        if a.cmd == "pack":
            print(pack(path, a.out), flush=True)
        elif a.cmd == "cvc":
            folder = Path(path) if Path(path).is_dir() else Path(path).parent
            target = Path(a.out or folder.parent) / f"{folder.name}{CVC_SUFFIX}"
            print(folder_to_cvc(folder, target, settings=not a.no_settings), flush=True)
        else:
            name = voice_name(path)
            print(unpack(path, Path(a.out) / name, name), flush=True)


if __name__ == "__main__":
    main()
