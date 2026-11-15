"""Voices as self-contained folders.

A voice is a folder holding two files, and nothing in either names a place on
the machine that made it:

    NAME-10000s/
      NAME-10000s.causalvc.json     {"decoder": "decoder.pt", "speaker": "NAME"}
      decoder.pt                    the voice's decoder, this one speaker only

Copy or zip the folder and it works anywhere this project is installed. The two
networks every voice shares (the content encoder and the pitch tracker) are not
in the folder: they come with the base that setup downloads.

    python -m causalvc.voice pack PATH [--out FOLDER]

turns an older voice file (one pointing at weights elsewhere on this machine)
into such a folder.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUFFIXES = (".causalvc.json", ".nextvc.json")


def shared(name: str) -> Path:
    """The content encoder ("encoder.pt") or pitch tracker ("pitch.pt") every voice uses."""
    from causalvc.weights import base_folder, ensure_base

    local = {"encoder.pt": "runs/models/enc-la2-460/encoder.pt", "pitch.pt": "runs/models/pitch-la2/pitch.pt"}[name]
    for path in (base_folder() / name, ROOT / local):  # the downloaded base, or where they were trained
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
    for suffix in SUFFIXES:
        name = name.removesuffix(suffix)
    return name


def load(description) -> dict:
    """A voice description file -> {"decoder", "pitch", "encoder", "speaker", "runtime"} with real paths.

    Relative paths are looked for beside the description first, then under the
    project folder the file names (older files), then under this one."""
    description = Path(description)
    spec = json.loads(description.read_text(encoding="utf-8"))
    bases = [description.parent] + ([Path(spec["project"])] if spec.get("project") else []) + [ROOT]
    decoder = existing(spec["decoder"], *bases)
    if decoder is None and Path(spec["decoder"]).is_absolute():  # written on another machine: try the file name here
        decoder = existing(Path(spec["decoder"]).name, description.parent)
    if decoder is None:
        raise FileNotFoundError(f"{description.name} points at {spec['decoder']}, which is not on this machine. "
                                "A voice folder should contain its own decoder.pt (python -m causalvc.voice pack).")
    return {
        "decoder": str(decoder),
        "pitch": str(existing(spec.get("pitch"), *bases) or shared("pitch.pt")),
        "encoder": str(existing(spec.get("encoder"), *bases)) if existing(spec.get("encoder"), *bases) else None,
        "speaker": spec["speaker"],
        "runtime": spec.get("runtime"),
    }


def export(decoder, speaker: str, folder, name: str | None = None) -> Path:
    """Write a self-contained voice folder from a decoder checkpoint; returns the description's path.

    Only `speaker` is kept from the checkpoint's speaker table, so the voices of
    the base it was built on do not travel with it."""
    import torch

    folder = Path(folder)
    name = name or folder.name
    ck = torch.load(decoder, map_location="cpu")
    index = ck["speakers"].index(speaker)
    generator = dict(ck["generator"])
    generator["speaker.weight"] = generator["speaker.weight"][index : index + 1].clone()
    folder.mkdir(parents=True, exist_ok=True)
    slim = {"generator": generator, "cfg": ck["cfg"], "speakers": [speaker], "step": ck.get("step", 0), "encoder": "encoder.pt"}
    torch.save(slim, folder / "decoder.pt.part")
    (folder / "decoder.pt.part").replace(folder / "decoder.pt")
    for old in folder.iterdir():  # one description per voice folder
        if old.name.endswith(SUFFIXES):
            old.unlink()
    target = folder / f"{name}.causalvc.json"
    target.write_text(json.dumps({"decoder": "decoder.pt", "speaker": speaker}, indent=1), encoding="utf-8")
    return target


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
    a = ap.parse_args()
    for path in a.path:
        print(pack(path, a.out), flush=True)


if __name__ == "__main__":
    main()
