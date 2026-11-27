"""Pretrained files the code needs, fetched on first use into weights/.

Each entry has a download address and, where known, the file's SHA-256. A file
already in weights/ is used as it is. CAUSALVC_WEIGHTS points somewhere else.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FOLDER = Path(os.environ.get("CAUSALVC_WEIGHTS", ROOT / "weights"))

FILES = {
    # The pitch model used to label training audio, from the same place Applio fetches it.
    "rmvpe.pt": (
        "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/predictors/rmvpe.pt",
        "6d62215f4306e3ca278246188607209f09af3dc77ed4232efdd069798c4ec193",
    ),
}


# The base a voice is built from: one zip holding these four files (checksums are of the files, not the zip).
BASE_URL = os.environ.get("CAUSALVC_BASE_URL", "https://huggingface.co/buckets/Mojobigbones/cvc/resolve/causalvc-base-v5-100hz.zip?download=true")
BASE_NAME = "causalvc-base"
BASE_FILES = {
    "last.pt": "beff4858e9020506e95a8a65770f37e527c2e5988be874671ac279d81310ad77",
    "encoder.pt": "4b7d7fe78fe01a94648eaf3216e2a14004da8bd09caf7ae0c6661a3251c26693",
    "pitch.pt": "a4c3f1098cd8592233203129059c93857d8575407edc05faee5b47af0b999636",
    "args.json": "5d4c6c958193497f29abf436d6900f699a4fae1b0845ebe0d6d1f7ce1932937b",
}

# Additional official pretrained bases selectable alongside Default, each downloaded on demand into weights/.
PRETRAINED_BASES: dict[str, tuple[str, str, dict[str, str]]] = {
    "v4": (
        "causalvc-base-v4",
        "https://huggingface.co/buckets/Mojobigbones/cvc/resolve/causalvc-base-v4-100hz.zip?download=true",
        {
            "last.pt": "e8f2fd8cf7fb61b68a454f73d1faea479d8a923aef7dd5242119a216d9ff6f61",
            "encoder.pt": "4b7d7fe78fe01a94648eaf3216e2a14004da8bd09caf7ae0c6661a3251c26693",
            "pitch.pt": "a4c3f1098cd8592233203129059c93857d8575407edc05faee5b47af0b999636",
            "args.json": "e39434d48669e6d63a659214df69be7a3f2378c17163e8e3b85dadc6d5f655b2",
        },
    ),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, label: str) -> None:
    """Fetch url to dest, through a .part file so an interrupted download is never mistaken for a whole one."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {label} from {url}", flush=True)
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "causalvc"})) as r, open(part, "wb") as f:
        total, got, shown = int(r.headers.get("Content-Length") or 0), 0, -1
        while chunk := r.read(1 << 20):
            f.write(chunk)
            got += len(chunk)
            pct = 100 * got // total if total else 0
            if pct // 10 != shown and total:
                shown = pct // 10
                print(f"  {pct}% of {total / 1e6:.0f} MB", flush=True)
    part.replace(dest)


def ensure(name: str) -> Path:
    """Path of a pretrained file, downloading it first if it is not there yet."""
    path = FOLDER / name
    if path.is_file():
        return path
    url, digest = FILES[name]
    download(url, path, name)
    if digest and sha256(path) != digest:
        path.unlink()
        sys.exit(f"{name}: the downloaded file does not match its expected checksum; nothing was kept")
    return path


def base_folder(name: str = "default") -> Path:
    if name == "default":
        return FOLDER / BASE_NAME
    if name in PRETRAINED_BASES:
        return FOLDER / PRETRAINED_BASES[name][0]
    raise KeyError(f"unknown pretrained base: {name}")


def _preserve_previous_base(folder: Path) -> None:
    """Before replacing weights/causalvc-base with a newer default, move an existing prior official base
    (e.g. v4) into its own subfolder so users keep it available without re-downloading."""
    args_path = folder / "args.json"
    if not args_path.is_file():
        return
    try:
        args_sha = sha256(args_path)
    except OSError:
        return
    for name, (sub, _url, files) in PRETRAINED_BASES.items():
        if args_sha == files["args.json"] and all((folder / f).is_file() for f in files):
            dest = FOLDER / sub
            if not dest.exists():
                print(f"preserving existing {name} base in {dest}", flush=True)
                shutil.move(str(folder), str(dest))
            return


def ensure_base(name: str = "default") -> Path:
    """Folder holding a pretrained base (last.pt, args.json, encoder.pt, pitch.pt), downloaded on first use."""
    if name == "default":
        folder, url, files, archive_name = base_folder(), BASE_URL, BASE_FILES, f"{BASE_NAME}.zip"
    elif name in PRETRAINED_BASES:
        sub, url, files = PRETRAINED_BASES[name]
        folder, archive_name = FOLDER / sub, f"{sub}.zip"
    else:
        sys.exit(f"unknown pretrained base '{name}'")

    if all((folder / f).is_file() for f in files) and (
        not url or sha256(folder / "args.json") == files["args.json"]
    ):
        return folder
    if not url:
        sys.exit(f"the base is not in {folder} and no download address is set (CAUSALVC_BASE_URL)")
    if name == "default":
        _preserve_previous_base(folder)
    archive = FOLDER / archive_name
    download(url, archive, archive_name)
    folder.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        # Take only the expected files, by name, wherever they sit in the archive; nothing else is written.
        members = {Path(m.filename).name: m for m in z.infolist() if not m.is_dir()}
        missing = [f for f in files if f not in members]
        if missing:
            sys.exit(f"{archive.name} does not contain {', '.join(missing)}")
        for fname, digest in files.items():
            print(f"unpacking {fname}", flush=True)
            with z.open(members[fname]) as src, open(folder / fname, "wb") as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
            if sha256(folder / fname) != digest:
                (folder / fname).unlink()
                sys.exit(f"{fname} in {archive.name} does not match its expected checksum; it was not kept")
    archive.unlink()
    return folder


#: What each app needs (scripts/setup.sh and setup.bat pass the app's name). Every voice runs on the
#: base's encoder and pitch tracker; RMVPE only labels training audio.
NEEDS = {"realtime": (), "training": tuple(FILES)}


if __name__ == "__main__":  # python -m causalvc.weights [realtime|training] : fetch them now (setup)
    app = sys.argv[1] if len(sys.argv) > 1 else "training"
    if app not in NEEDS:
        sys.exit(f"usage: python -m causalvc.weights [{'|'.join(NEEDS)}]")
    for name in NEEDS[app]:
        print(ensure(name))
    print(ensure_base())
