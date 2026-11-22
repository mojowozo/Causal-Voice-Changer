"""The updater: version rules, release parsing, safe unpacking, and installing over a project folder."""

import importlib.util
import io
import json
import tarfile
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from causalvc import update

SCRIPT = Path(update.ROOT) / "scripts" / "apply_update.py"
spec = importlib.util.spec_from_file_location("apply_update", SCRIPT)
apply_update = importlib.util.module_from_spec(spec)
spec.loader.exec_module(apply_update)

RELEASE_JSON = {
    "tag_name": "v0.4.0",
    "name": "v0.4.0 Release",
    "html_url": "https://github.com/x/y/releases/tag/v0.4.0",
    "body": "# V0.4.0\r\n- better",
    "assets": [
        {"name": "Causal-Voice-Changer-v0.4.0-linux-x86_64.tar.gz", "size": 10,
         "browser_download_url": "https://example/l.tgz", "digest": "sha256:" + "a" * 64},
        {"name": "Causal-Voice-Changer-v0.4.0-windows-x64.zip", "size": 20, "browser_download_url": "https://example/w.zip"},
    ],
}


class VersionRules(unittest.TestCase):
    def test_compare(self):
        self.assertTrue(update.is_newer("v0.4.0", "v0.3.0"))
        self.assertTrue(update.is_newer("v0.10.0", "v0.9.9"))
        self.assertFalse(update.is_newer("v0.3.0", "v0.3.0"))
        self.assertFalse(update.is_newer("v0.2.9", "v0.3.0"))
        self.assertTrue(update.is_newer("v0.3.1", "unknown"))
        self.assertFalse(update.is_newer("nightly", "v0.3.0"))

    def test_release_info_picks_this_systems_download(self):
        linux = update.release_info(RELEASE_JSON, "-linux-x86_64.tar.gz")
        self.assertEqual(linux["asset"]["url"], "https://example/l.tgz")
        self.assertEqual(linux["asset"]["sha256"], "a" * 64)
        self.assertEqual(linux["notes"], "V0.4.0\n• better")
        win = update.release_info(RELEASE_JSON, "-windows-x64.zip")
        self.assertEqual(win["asset"]["sha256"], "")
        self.assertIsNone(update.release_info(RELEASE_JSON, "-macos.zip")["asset"])


class ReleaseNotes(unittest.TestCase):
    def test_markdown_becomes_readable_text(self):
        notes = ("## What's new\r\n"
                 "* Added the ability to go down to 10ms block size\r\n"
                 "* A **pitch floor** (defaults to *off*)\r\n"
                 "  - nested, see [the docs](https://example/docs) and `pitch_floor_hz`\r\n"
                 "- Better handling for low voices\r\n"
                 "---\r\n\r\n\r\n"
                 "No retraining is needed: 2*3*4 stays.")
        self.assertEqual(update.notes_text(notes),
                         "What's new\n"
                         "• Added the ability to go down to 10ms block size\n"
                         "• A pitch floor (defaults to off)\n"
                         "  • nested, see the docs and pitch_floor_hz\n"
                         "• Better handling for low voices\n\n"
                         "No retraining is needed: 2*3*4 stays.")


def fake_project(root: Path, version: str, files: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "VERSION").write_text(version + "\n")
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    names = ["VERSION", *files]
    (root / "MANIFEST.txt").write_text("\n".join(sorted([*names, "MANIFEST.txt"])) + "\n")


class Unpacking(unittest.TestCase):
    def test_tar_and_zip_strip_the_top_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            tgz, zp = tmp / "r.tar.gz", tmp / "r.zip"
            with tarfile.open(tgz, "w:gz") as t:
                for name, data in (("CVC-v1/run-realtime.sh", b"#!"), ("CVC-v1/a/b.txt", b"x")):
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    t.addfile(info, io.BytesIO(data))
            with zipfile.ZipFile(zp, "w") as z:
                z.writestr("CVC-v1/run-realtime.sh", "#!")
            self.assertEqual((update.unpack(tgz, tmp / "t") / "a" / "b.txt").read_text(), "x")
            self.assertTrue((update.unpack(zp, tmp / "z") / "run-realtime.sh").is_file())

    def test_refuses_paths_outside(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for bad in ("../evil.txt", "/etc/evil", "C:/evil"):
                zp = tmp / "bad.zip"
                with zipfile.ZipFile(zp, "w") as z:
                    z.writestr("CVC/run-realtime.sh", "#!")
                    z.writestr(bad, "x")
                with self.assertRaises(RuntimeError):
                    update.unpack(zp, tmp / "out")
                self.assertFalse((tmp / "evil.txt").exists())

    def test_refuses_something_that_is_not_a_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            zp = Path(tmp) / "r.zip"
            with zipfile.ZipFile(zp, "w") as z:
                z.writestr("x/readme.txt", "hi")
            with self.assertRaises(RuntimeError):
                update.unpack(zp, Path(tmp) / "out")


class Installing(unittest.TestCase):
    def test_copies_keeps_user_files_and_deletes_dropped_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, staged = Path(tmp) / "app", Path(tmp) / "new"
            fake_project(root, "v0.3.0", {"run-realtime.sh": "old", "gone.py": "old", "realtime/models/README.md": "old",
                                          "keep/dir/only.py": "old"})
            (root / ".venv").mkdir()
            (root / ".venv" / "python").write_text("env")
            (root / "realtime/models/Voice/voice.cvc").parent.mkdir(parents=True)
            (root / "realtime/models/Voice/voice.cvc").write_text("mine")
            (root / "update-settings.json").write_text('{"skipped": "v0.3.5"}')
            fake_project(staged, "v0.4.0", {"run-realtime.sh": "new", "new.py": "new", "realtime/models/README.md": "new",
                                            "realtime/configs/default.json": "{}", "keep/dir/only.py": "new"})
            result = apply_update.install(root, staged, lock_seconds=0)
            self.assertTrue(result["ok"], result)
            self.assertEqual((root / "VERSION").read_text().strip(), "v0.4.0")
            self.assertEqual((root / "run-realtime.sh").read_text(), "new")
            self.assertEqual((root / "new.py").read_text(), "new")
            self.assertFalse((root / "gone.py").exists())  # in the old manifest, not the new one
            self.assertEqual((root / ".venv/python").read_text(), "env")
            self.assertEqual((root / "realtime/models/Voice/voice.cvc").read_text(), "mine")
            self.assertEqual((root / "realtime/models/README.md").read_text(), "old")  # protected: kept
            self.assertTrue((root / "realtime/configs/default.json").is_file())  # protected but missing: added
            self.assertIn("v0.3.5", (root / "update-settings.json").read_text())
            self.assertFalse(list(root.rglob("*.updating")))

    def test_no_old_manifest_deletes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, staged = Path(tmp) / "app", Path(tmp) / "new"
            fake_project(root, "v0.3.0", {"gone.py": "old"})
            (root / "MANIFEST.txt").unlink()
            fake_project(staged, "v0.4.0", {"run-realtime.sh": "new"})
            self.assertTrue(apply_update.install(root, staged, lock_seconds=0)["ok"])
            self.assertTrue((root / "gone.py").exists())

    def test_a_locked_file_changes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, staged = Path(tmp) / "app", Path(tmp) / "new"
            fake_project(root, "v0.3.0", {"app.exe": "old", "a.py": "old"})
            fake_project(staged, "v0.4.0", {"app.exe": "new", "a.py": "new"})
            real = apply_update.writable
            with mock.patch.object(apply_update, "writable", lambda p: p.name != "app.exe" and real(p)):
                result = apply_update.install(root, staged, lock_seconds=0)
            self.assertFalse(result["ok"])
            self.assertIn("app.exe", result["message"])
            self.assertEqual((root / "a.py").read_text(), "old")
            self.assertEqual((root / "VERSION").read_text().strip(), "v0.3.0")


class UpdaterStates(unittest.TestCase):
    def make(self, tmp, current="v0.3.0", git=False, settings=None):
        root = Path(tmp)
        (root / "VERSION").write_text(current)
        if git:
            (root / ".git").mkdir()
        if settings:
            (root / "update-settings.json").write_text(json.dumps(settings))
        events = []
        done = threading.Event()

        def emit(name, data):
            events.append(data)
            if data["state"] not in ("checking",):
                done.set()

        with mock.patch.object(update, "current_version", lambda r: current):
            u = update.Updater("realtime", emit, root)
        return u, events, done

    def run_check(self, u, done, manual, release=None, error=None):
        done.clear()
        rel = update.release_info(release or RELEASE_JSON, "-linux-x86_64.tar.gz")
        side = mock.Mock(side_effect=error) if error else mock.Mock(return_value=rel)
        with mock.patch.object(update, "latest", side):
            u.check(manual)
            self.assertTrue(done.wait(5))

    def test_launch_check_offers_a_newer_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            u, events, done = self.make(tmp)
            self.run_check(u, done, manual=False)
            self.assertEqual(events[-1]["state"], "available")
            self.assertEqual(events[-1]["version"], "v0.4.0")

    def test_skipped_version_is_quiet_at_launch_but_shown_when_asked(self):
        with tempfile.TemporaryDirectory() as tmp:
            u, events, done = self.make(tmp, settings={"skipped": "v0.4.0"})
            self.run_check(u, done, manual=False)
            self.assertEqual(events[-1]["state"], "idle")
            self.run_check(u, done, manual=True)
            self.assertEqual(events[-1]["state"], "available")

    def test_skip_and_toggle_are_remembered(self):
        with tempfile.TemporaryDirectory() as tmp:
            u, events, done = self.make(tmp)
            u.skip("v0.4.0")
            u.set_check_at_launch(False)
            saved = json.loads((Path(tmp) / "update-settings.json").read_text())
            self.assertEqual(saved, {"check_at_launch": False, "skipped": "v0.4.0"})
            u2, _, _ = self.make(tmp)
            self.assertFalse(u2.snapshot()["check_at_launch"])
            with mock.patch.object(u2, "check") as check:
                u2.launch()
                check.assert_not_called()

    def test_up_to_date_and_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            u, events, done = self.make(tmp, current="v0.4.0")
            self.run_check(u, done, manual=False)
            self.assertEqual(events[-1]["state"], "idle")
            self.run_check(u, done, manual=True)
            self.assertEqual(events[-1]["state"], "none")
            self.run_check(u, done, manual=False, error=OSError("offline"))
            self.assertEqual(events[-1]["state"], "idle")
            self.run_check(u, done, manual=True, error=OSError("offline"))
            self.assertEqual(events[-1]["state"], "error")
            self.assertIn("offline", events[-1]["error"])

    def test_git_checkout_never_downloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            u, events, done = self.make(tmp, git=True)
            self.run_check(u, done, manual=True)
            self.assertTrue(events[-1]["git"])
            with self.assertRaises(update.UpdateError):
                u.download()

    def test_a_failed_install_is_reported_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / ".update").mkdir()
            (Path(tmp) / ".update" / "failed.json").write_text(json.dumps({"message": "files in use"}))
            u, _, _ = self.make(tmp)
            self.assertEqual(u.snapshot()["state"], "failed")
            self.assertEqual(u.snapshot()["error"], "files in use")
            self.assertEqual(self.make(tmp)[0].snapshot()["state"], "idle")


if __name__ == "__main__":
    unittest.main()
