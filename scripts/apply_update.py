"""Install an unpacked release over the project folder, then start the app again.

Started by causalvc.update (from a copy in .update/, so it never replaces itself) once a
release has been downloaded:

    python -I apply_update.py --root PROJECT --staged UNPACKED_RELEASE --app realtime|training --wait PID,PID

1. Waits for the app's window and its Python process to exit.
2. Makes sure every file it will replace can be written (on Windows a running program's
   files cannot). If one stays locked, nothing is changed: it writes .update/failed.json,
   which the app shows on its next start, and starts the app again as it was.
3. Copies the release's files over the project. What the user made or installed is kept:
   the Python environments, voices, builds, settings and downloaded weights are never
   overwritten or deleted (a file the release has there is only added when missing).
4. Deletes files the old release had (its MANIFEST.txt) that the new one no longer has.
5. On Linux, updates the Python packages when uv.lock changed (the Windows launchers
   always do this themselves).
6. Starts the app again through its launcher, which rebuilds the window if needed.

Standard library only: it runs with the project's Python while the project's own files change.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

WINDOWS = sys.platform.startswith("win")
# Kept as they are: made or installed on this machine. Release files there are only added when missing.
PROTECTED = (".venv", ".venv-win", ".dotnet", ".git", ".update", "runs", "data", "weights", "logs",
             "third_party", "realtime/models", "realtime/configs", "realtime/logs", "update-settings.json")
WAIT_SECONDS = 120
LOCK_SECONDS = 30


def log(*parts) -> None:
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def protected(rel: str) -> bool:
    rel = rel.replace("\\", "/")
    return any(rel == p or rel.startswith(p + "/") for p in PROTECTED)


def alive(pid: int) -> bool:
    if WINDOWS:
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x00100000 | 0x1000, False, pid)  # SYNCHRONIZE | QUERY_LIMITED
        if not handle:
            return False
        try:
            return ctypes.windll.kernel32.WaitForSingleObject(handle, 0) == 0x102  # WAIT_TIMEOUT: still running
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:  # a zombie (exited, not yet reaped by its parent) is gone for our purposes
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return True


def wait_for(pids: list[int], seconds: float = WAIT_SECONDS) -> None:
    end = time.monotonic() + seconds
    while any(alive(p) for p in pids) and time.monotonic() < end:
        time.sleep(0.25)
    still = [p for p in pids if alive(p)]
    if still:
        log("still running after", seconds, "s:", still, "- going on; locked files are checked next")


def release_files(staged: Path) -> list[str]:
    return sorted(p.relative_to(staged).as_posix() for p in staged.rglob("*") if p.is_file())


def manifest(folder: Path) -> set[str] | None:
    try:
        return {line.strip() for line in (folder / "MANIFEST.txt").read_text(encoding="utf-8").splitlines() if line.strip()}
    except OSError:
        return None


def writable(path: Path) -> bool:
    if not path.exists():
        return True
    try:
        with open(path, "r+b"):
            return True
    except PermissionError:
        return False
    except OSError:
        return False


def plan(root: Path, staged: Path) -> tuple[list[str], list[str]]:
    """(files to copy, files to delete)."""
    files = release_files(staged)
    copy = [f for f in files if not (protected(f) and (root / f).exists())]
    old, new = manifest(root), manifest(staged)
    delete = []
    if old is not None and new is not None:
        delete = sorted(f for f in old - new if not protected(f) and (root / f).is_file())
    return copy, delete


def locked(root: Path, files: list[str], seconds: float = LOCK_SECONDS) -> list[str]:
    end = time.monotonic() + seconds
    while True:
        busy = [f for f in files if not writable(root / f)]
        if not busy or time.monotonic() >= end:
            return busy
        time.sleep(1)


def install(root: Path, staged: Path, lock_seconds: float = LOCK_SECONDS) -> dict:
    """Copy the release in; returns {"ok": bool, "message": str, "copied": n, "deleted": n}."""
    copy, delete = plan(root, staged)
    busy = locked(root, copy + delete, lock_seconds)
    if busy:
        names = ", ".join(busy[:3]) + (" …" if len(busy) > 3 else "")
        return {"ok": False, "copied": 0, "deleted": 0,
                "message": f"Some files were in use ({names}). Close every Causal Voice Changer window "
                           "(the voice changer and the training app) and click Update again."}
    lock_before = (root / "uv.lock").read_bytes() if (root / "uv.lock").is_file() else b""
    for f in copy:
        dest = root / f
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".updating")
        shutil.copy2(staged / f, tmp)
        os.replace(tmp, dest)
    for f in delete:
        (root / f).unlink(missing_ok=True)
        parent = (root / f).parent
        while parent != root and parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent
    lock_after = (root / "uv.lock").read_bytes() if (root / "uv.lock").is_file() else b""
    log(f"copied {len(copy)} files, deleted {len(delete)}")
    return {"ok": True, "copied": len(copy), "deleted": len(delete), "message": "",
            "deps_changed": lock_before != lock_after}


def find_uv() -> str | None:
    found = shutil.which("uv")
    if found:
        return found
    for p in (Path.home() / ".local/bin/uv", Path.home() / ".cargo/bin/uv"):
        if p.is_file():
            return str(p)
    return None


def relaunch(root: Path, app: str) -> None:
    name = {"realtime": "run-realtime", "training": "run-training"}[app]
    if WINDOWS:
        subprocess.Popen(["cmd.exe", "/c", str(root / f"{name}.bat")], cwd=str(root),
                         creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:
        out = open(root / ".update" / "relaunch.log", "w")
        subprocess.Popen(["bash", str(root / f"{name}.sh")], cwd=str(root), start_new_session=True,
                         stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--staged", required=True)
    ap.add_argument("--app", choices=("realtime", "training"), required=True)
    ap.add_argument("--wait", default="")
    ap.add_argument("--no-relaunch", action="store_true")
    a = ap.parse_args(argv)
    root, staged = Path(a.root).resolve(), Path(a.staged).resolve()
    pids = [int(p) for p in a.wait.split(",") if p.strip().isdigit()]
    log("installing", staged, "over", root, "for", a.app, "after", pids)
    wait_for(pids)
    try:
        result = install(root, staged)
    except Exception as error:  # a disk error mid-copy: say so, and keep the download to retry
        result = {"ok": False, "message": f"The update stopped part way: {error}. Click Update again to finish it."}
    if not result["ok"]:
        log("not installed:", result["message"])
        (root / ".update" / "failed.json").write_text(json.dumps({"message": result["message"]}), encoding="utf-8")
    else:
        if result.get("deps_changed") and not WINDOWS:
            uv = find_uv()
            if uv:
                log("updating the Python packages")
                subprocess.run([uv, "sync"], cwd=str(root))
            else:
                log("uv not found; run ./run-training.sh once to update the Python packages")
        # The unpacked download (causalvc.update unpacks into .update/release-<tag>/<top folder>).
        shutil.rmtree(staged.parent if staged.parent.name.startswith("release-") else staged, ignore_errors=True)
    if not a.no_relaunch:
        relaunch(root, a.app)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
