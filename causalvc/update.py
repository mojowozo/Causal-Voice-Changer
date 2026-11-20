"""Checking GitHub for a newer release, downloading it, and handing it to scripts/apply_update.py.

Shared by the realtime engine and the trainer: each owns one ``Updater`` and reports its
progress through ``emit("update", snapshot)``. Nothing here imports torch, so it works even
when the rest of the project fails to start (an update may be what fixes that).

A release is the whole project folder (see .github/workflows/app-builds.yml), so an update
copies the new files over this folder, keeping everything the user made or installed.
The only thing sent to GitHub is an anonymous request for the latest release.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent
REPO = os.environ.get("CAUSALVC_UPDATE_REPO", "mojowozo/Causal-Voice-Changer")
API = os.environ.get("CAUSALVC_UPDATE_API", "https://api.github.com")
SETTINGS = ROOT / "update-settings.json"
WORK = ROOT / ".update"  # downloads, the unpacked release, the installer's log
FAILED = WORK / "failed.json"  # written by apply_update.py when it could not install
TIMEOUT = 10
NOTES_CHARS = 2000
DEFAULTS = {"check_at_launch": True, "skipped": ""}


def asset_suffix(platform: str = sys.platform) -> str | None:
    """The release download for this system, by the end of its name."""
    if platform.startswith("win"):
        return "-windows-x64.zip"
    if platform.startswith("linux"):
        return "-linux-x86_64.tar.gz"
    return None


def parse_version(text: str) -> tuple[int, int, int] | None:
    m = re.match(r"^v?(\d+)\.(\d+)(?:\.(\d+))?", (text or "").strip())
    return (int(m[1]), int(m[2]), int(m[3] or 0)) if m else None


def is_newer(candidate: str, current: str) -> bool:
    new, old = parse_version(candidate), parse_version(current)
    return new is not None and (old is None or new > old)


def is_git_checkout(root: Path = ROOT) -> bool:
    return (root / ".git").exists()


def current_version(root: Path = ROOT) -> str:
    """The release this folder is: the VERSION file a release download carries, or in a git
    checkout the newest tag reachable from what is checked out (VERSION may lag behind there)."""
    if is_git_checkout(root):
        try:
            tag = subprocess.run(["git", "describe", "--tags", "--abbrev=0"], cwd=root, capture_output=True,
                                 text=True, timeout=5).stdout.strip()
            if parse_version(tag):
                return tag
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        return (root / "VERSION").read_text(encoding="utf-8").strip() or "unknown"
    except OSError:
        return "unknown"


def load_settings(path: Path = SETTINGS) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    out = dict(DEFAULTS)
    if isinstance(data.get("check_at_launch"), bool):
        out["check_at_launch"] = data["check_at_launch"]
    if isinstance(data.get("skipped"), str):
        out["skipped"] = data["skipped"]
    return out


def save_settings(settings: dict, path: Path = SETTINGS) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _get(url: str, accept: str = "application/vnd.github+json"):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "causalvc", "Accept": accept}),
                                  timeout=TIMEOUT)


def release_info(data: dict, suffix: str | None = None) -> dict:
    """What we need from GitHub's release JSON; ``asset`` is None when there is no download for this system."""
    suffix = asset_suffix() if suffix is None else suffix
    asset = None
    for a in data.get("assets") or []:
        if suffix and isinstance(a.get("name"), str) and a["name"].endswith(suffix):
            digest = a.get("digest") or ""
            asset = {"name": a["name"], "url": a.get("browser_download_url"), "size": int(a.get("size") or 0),
                     "sha256": digest[7:] if digest.startswith("sha256:") else ""}
            break
    notes = (data.get("body") or "").replace("\r\n", "\n").strip()
    if len(notes) > NOTES_CHARS:
        notes = notes[:NOTES_CHARS].rsplit("\n", 1)[0] + "\n…"
    return {"version": data.get("tag_name") or "", "name": data.get("name") or "", "notes": notes,
            "url": data.get("html_url") or f"https://github.com/{REPO}/releases", "asset": asset}


def latest() -> dict:
    """The newest published release (drafts and pre-releases are never offered)."""
    with _get(f"{API}/repos/{REPO}/releases/latest") as r:
        return release_info(json.loads(r.read().decode("utf-8")))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(asset: dict, dest_dir: Path, progress=lambda done, total: None) -> Path:
    """Fetch a release download through a .part file and check its size and checksum."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / asset["name"]
    part = dest.with_name(dest.name + ".part")
    with _get(asset["url"], accept="application/octet-stream") as r, open(part, "wb") as f:
        total, got = int(r.headers.get("Content-Length") or asset.get("size") or 0), 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            got += len(chunk)
            progress(got, total)
    if asset.get("size") and part.stat().st_size != asset["size"]:
        part.unlink()
        raise RuntimeError("The download was incomplete. Try again.")
    if asset.get("sha256") and sha256(part) != asset["sha256"]:
        part.unlink()
        raise RuntimeError("The download does not match the release's checksum. Try again.")
    part.replace(dest)
    return dest


def _safe_member(name: str) -> bool:
    p = PurePosixPath(name.replace("\\", "/"))
    return bool(name) and not p.is_absolute() and ".." not in p.parts and not re.match(r"^[A-Za-z]:", name)


def unpack(archive: Path, dest: Path) -> Path:
    """Extract a release archive into ``dest`` and return the project folder inside it.
    Refuses archives with paths that would land outside ``dest``."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            bad = [n for n in z.namelist() if not _safe_member(n)]
            if bad:
                raise RuntimeError(f"The download holds an unsafe path ({bad[0]}); nothing was installed.")
            z.extractall(dest)
    else:
        with tarfile.open(archive) as t:
            bad = [m.name for m in t.getmembers() if not _safe_member(m.name) or m.issym() or m.islnk() or m.isdev()]
            if bad:
                raise RuntimeError(f"The download holds an unsafe path ({bad[0]}); nothing was installed.")
            t.extractall(dest, filter="data")
    entries = [p for p in dest.iterdir()]
    top = entries[0] if len(entries) == 1 and entries[0].is_dir() else dest
    if not (top / "run-realtime.sh").is_file():
        raise RuntimeError("The download is not a Causal Voice Changer release; nothing was installed.")
    return top


def launcher(app: str, root: Path = ROOT, platform: str = sys.platform) -> Path:
    name = {"realtime": "run-realtime", "training": "run-training"}[app]
    return root / (name + (".bat" if platform.startswith("win") else ".sh"))


def start_installer(staged: Path, app: str, wait_pids: list[int], root: Path = ROOT) -> None:
    work = root / WORK.name
    """Start scripts/apply_update.py on its own, from a copy outside the files it replaces;
    it waits for ``wait_pids`` to exit, installs ``staged`` over ``root`` and starts ``app`` again."""
    work.mkdir(exist_ok=True)
    script = work / "apply_update.py"
    shutil.copy2(staged / "scripts" / "apply_update.py", script)  # the new release's installer
    args = [sys.executable, "-I", str(script), "--root", str(root), "--staged", str(staged), "--app", app,
            "--wait", ",".join(str(p) for p in wait_pids)]
    log = open(work / "last-update.log", "w", encoding="utf-8")
    kw = dict(cwd=str(work), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, close_fds=True)
    if sys.platform.startswith("win"):
        kw["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    else:
        kw["start_new_session"] = True
    subprocess.Popen(args, **kw)
    log.close()


class Updater:
    """One app's update state. ``emit("update", snapshot)`` after every change.

    States: idle, checking, none (up to date), available, downloading, ready (downloaded and
    unpacked), installing (the app should now close), error, failed (the last install did not finish).
    A check at launch reports only a newer version that was not skipped; a manual check reports
    every outcome. In a git checkout an update is never installed: the window says to use git pull.
    """

    def __init__(self, app: str, emit, root: Path = ROOT):
        self.app, self.emit, self.root = app, emit, root
        self.lock = threading.Lock()
        self.work, self.settings_path = root / WORK.name, root / SETTINGS.name
        self.settings = load_settings(self.settings_path)
        self.release: dict | None = None
        self.staged: Path | None = None
        self.manual = False
        self.state = {"state": "idle", "current": current_version(root), "git": is_git_checkout(root),
                      "check_at_launch": self.settings["check_at_launch"], "version": "", "name": "", "notes": "",
                      "url": "", "progress": 0.0, "error": "", "manual": False}
        failed = self.work / FAILED.name
        if failed.is_file():
            try:
                why = json.loads(failed.read_text(encoding="utf-8")).get("message", "")
            except (OSError, ValueError):
                why = ""
            failed.unlink(missing_ok=True)
            self.state.update(state="failed", error=why or "The last update did not finish.")

    def snapshot(self) -> dict:
        with self.lock:
            return dict(self.state)

    def _set(self, **changes) -> None:
        with self.lock:
            self.state.update(changes)
            snap = dict(self.state)
        self.emit("update", snap)

    def launch(self) -> None:
        """At start: check in the background when the user wants that."""
        if self.settings["check_at_launch"] and self.state["state"] == "idle":
            self.check(manual=False)

    def check(self, manual: bool = True) -> None:
        if self.state["state"] in ("checking", "downloading", "installing"):
            return
        self._set(state="checking", manual=manual, error="")
        threading.Thread(target=self._check, args=(manual,), daemon=True).start()

    def _check(self, manual: bool) -> None:
        try:
            rel = latest()
        except Exception as error:  # offline, rate-limited, GitHub down: only a manual check says so
            self._set(state="error" if manual else "idle", error=f"Could not check for updates: {error}")
            return
        self.release = rel
        newer = is_newer(rel["version"], self.state["current"])
        if not newer or (not manual and rel["version"] == self.settings["skipped"]):
            self._set(state="none" if manual else "idle", version=rel["version"], url=rel["url"])
            return
        if rel["asset"] is None and not self.state["git"]:
            self._set(state="error" if manual else "idle", version=rel["version"], url=rel["url"],
                      error="This release has no download for this system.")
            return
        self._set(state="available", version=rel["version"], name=rel["name"], notes=rel["notes"], url=rel["url"])

    def skip(self, version: str) -> None:
        self.settings["skipped"] = version
        save_settings(self.settings, self.settings_path)
        self._set(state="idle")

    def later(self) -> None:
        """Ask me next time: nothing is remembered."""
        self._set(state="idle")

    def set_check_at_launch(self, on: bool) -> None:
        self.settings["check_at_launch"] = bool(on)
        save_settings(self.settings, self.settings_path)
        self._set(check_at_launch=bool(on))

    def download(self) -> None:
        if self.state["git"]:
            raise UpdateError("git_checkout", "This folder is a git checkout: update it with git pull.")
        if self.release is None or self.release["asset"] is None or self.state["state"] not in ("available", "error"):
            raise UpdateError("no_update", "There is no update to download.")
        self._set(state="downloading", progress=0.0, error="")
        threading.Thread(target=self._download, daemon=True).start()

    def _download(self) -> None:
        rel = self.release
        last = [-1]

        def progress(done, total):
            pct = int(100 * done / total) if total else 0
            if pct != last[0]:
                last[0] = pct
                self._set(progress=pct / 100)

        try:
            archive = download(rel["asset"], self.work, progress)
            self.staged = unpack(archive, self.work / f"release-{rel['version']}")
            archive.unlink(missing_ok=True)
        except Exception as error:
            self._set(state="error", error=f"The update could not be downloaded: {error}")
            return
        self._set(state="ready", progress=1.0)

    def install(self, window_pid: int | None) -> None:
        """Start the installer; the app must close right after (the installer waits for it)."""
        if self.state["state"] != "ready" or self.staged is None:
            raise UpdateError("not_ready", "The update has not been downloaded yet.")
        pids = [os.getpid()] + ([window_pid] if window_pid else [])
        start_installer(self.staged, self.app, pids, self.root)
        self._set(state="installing")


class UpdateError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
