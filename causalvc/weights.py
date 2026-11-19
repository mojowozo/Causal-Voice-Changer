"""Pretrained files the code needs, fetched on first use into weights/.

Each entry has a download address and, where known, the file's SHA-256. A file
already in weights/ is used as it is. CAUSALVC_WEIGHTS points somewhere else.
"""

from __future__ import annotations

import hashlib
import os
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
BASE_URL = os.environ.get("CAUSALVC_BASE_URL", "https://huggingface.co/buckets/Mojobigbones/cvc/resolve/causalvc-base-v3-100hz.zip?download=true")
BASE_NAME = "causalvc-base"
BASE_FILES = {
    "last.pt": "b3d901474d00be83f614190a4ebf1a339786848d777f222caab7caee761edbcb",
    "encoder.pt": "7ade63f91e37c9700a229cfb0265b569273b02c14de70bbb9db19cc8618494c3",
    "pitch.pt": "b946c0d73d07c6d3e62136193e9779527501590b0e88b508b436b8955ea0353b",
    "args.json": "0af6610fc33225f223f04d715fa36aa76c8a5776c2c6c1b536b4c0e7af8e35b4",
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


def base_folder() -> Path:
    return FOLDER / BASE_NAME


def ensure_base() -> Path:
    """Folder holding the base (last.pt, args.json, encoder.pt, pitch.pt), downloaded and unpacked on first use."""
    folder = base_folder()
    if all((folder / name).is_file() for name in BASE_FILES) and (
        not BASE_URL or sha256(folder / "args.json") == BASE_FILES["args.json"]
    ):
        return folder
    if not BASE_URL:
        sys.exit(f"the base is not in {folder} and no download address is set (CAUSALVC_BASE_URL)")
    archive = FOLDER / f"{BASE_NAME}.zip"
    download(BASE_URL, archive, f"{BASE_NAME}.zip")
    folder.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        # Take only the expected files, by name, wherever they sit in the archive; nothing else is written.
        members = {Path(m.filename).name: m for m in z.infolist() if not m.is_dir()}
        missing = [name for name in BASE_FILES if name not in members]
        if missing:
            sys.exit(f"{archive.name} does not contain {', '.join(missing)}")
        for name, digest in BASE_FILES.items():
            print(f"unpacking {name}", flush=True)
            with z.open(members[name]) as src, open(folder / name, "wb") as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
            if sha256(folder / name) != digest:
                (folder / name).unlink()
                sys.exit(f"{name} in {archive.name} does not match its expected checksum; it was not kept")
    archive.unlink()
    return folder


if __name__ == "__main__":  # python -m causalvc.weights : fetch everything now (setup)
    for name in FILES:
        print(ensure(name))
    print(ensure_base())
