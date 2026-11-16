"""Voices as single .cvc files: installing, exporting, and what is refused."""

import importlib.util
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import torch

from engine import core
from tests.test_model_library import ModelFolders
from tools.model_import import ModelImportError, import_models
from tools.model_registry import discover_models
from tools.model_import import voice_files


REPOSITORY = Path(__file__).resolve().parents[2]


def checkpoint(path, speakers):
    """A tiny decoder checkpoint with one embedding row per speaker."""
    weights = torch.arange(len(speakers) * 4, dtype=torch.float32).reshape(len(speakers), 4)
    torch.save({"generator": {"speaker.weight": weights, "body": torch.ones(3)}, "cfg": {"dim": 4},
                "speakers": list(speakers), "step": 7}, path)
    return Path(path)


def decoder_in(cvc):
    with zipfile.ZipFile(cvc) as archive:
        return torch.load(io.BytesIO(archive.read("decoder.pt")), map_location="cpu")


def make_cvc(path, speaker="Alto", decoder=b"weights", settings=None):
    return voice_files.write_cvc(path, Path(path).stem, speaker, decoder, settings)


class InstallTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.models = self.root / "models"

    def test_a_dropped_cvc_is_copied_in_as_it_is(self):
        cvc = make_cvc(self.root / "Alto-1000s.cvc", settings=b'{"pitch": 3}')

        names = import_models([str(cvc)], self.models)

        self.assertEqual(names, ["Alto-1000s"])
        self.assertEqual([p.relative_to(self.models).as_posix() for p in self.models.rglob("*")],
                         ["Alto", "Alto/Alto-1000s.cvc"])
        self.assertEqual((self.models / "Alto" / "Alto-1000s.cvc").read_bytes(), cvc.read_bytes())
        (entry,) = discover_models(self.models)
        self.assertEqual(entry.size_bytes, cvc.stat().st_size)
        self.assertTrue(cvc.exists())  # a dropped file is copied, not moved

    def test_a_second_copy_gets_a_suffix(self):
        cvc = make_cvc(self.root / "Alto.cvc")
        import_models([str(cvc)], self.models)
        self.assertEqual(import_models([str(cvc)], self.models), ["Alto (2)"])

    def test_files_that_are_not_voices_are_refused_and_leave_nothing(self):
        (self.root / "fake.cvc").write_bytes(b"not a zip")
        with zipfile.ZipFile(self.root / "other.cvc", "w") as archive:
            archive.writestr("voice.json", '{"format": "something-else"}')
            archive.writestr("decoder.pt", b"x")
        for name in ("fake.cvc", "other.cvc"):
            with self.assertRaises(ModelImportError) as raised:
                import_models([str(self.root / name)], self.models)
            self.assertEqual(raised.exception.code, "import_failed")
        self.assertEqual([p.name for p in self.models.iterdir()], [])

    def test_only_the_known_members_are_unpacked(self):
        path = self.root / "Evil.cvc"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("voice.json", json.dumps({"format": "causalvc-voice", "version": 1, "speaker": "x"}))
            archive.writestr("decoder.pt", b"x")
            archive.writestr("../escaped.txt", b"x")
            archive.writestr("run.bat", b"x")
        voice_files.unpack(path, self.root / "unpacked")
        self.assertEqual(sorted(p.name for p in (self.root / "unpacked").iterdir()), ["decoder.pt", "unpacked.causalvc.json"])
        self.assertFalse((self.root / "escaped.txt").exists())

    def test_newer_formats_are_refused_with_a_reason(self):
        path = self.root / "Future.cvc"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("voice.json", json.dumps({"format": "causalvc-voice", "version": 99, "speaker": "x"}))
            archive.writestr("decoder.pt", b"x")
        with self.assertRaises(ModelImportError) as raised:
            import_models([str(path)], self.models)
        self.assertIn("newer version", raised.exception.message)

    def test_a_cvc_put_in_the_models_folder_by_hand_is_a_voice_as_it_is(self):
        self.models.mkdir()
        make_cvc(self.models / "Bass.cvc", speaker="Bass")

        self.assertEqual([entry.name for entry in discover_models(self.models)], ["Bass"])
        self.assertEqual([p.name for p in self.models.iterdir()], ["Bass.cvc"])


class CacheTest(unittest.TestCase):
    """A .cvc is loaded through a cache outside the project, which cleans up after itself."""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.cache = self.root / "cache"
        patcher = mock.patch.object(voice_files, "cache_folder", return_value=self.cache)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_renamed_replaced_and_deleted_files_leave_no_cache_behind(self):
        alto = make_cvc(self.root / "Alto.cvc")
        first = voice_files.cached(alto).parent
        self.assertTrue((first / "decoder.pt").is_file())
        self.assertEqual(voice_files.cached(alto).parent, first)  # unpacked once

        make_cvc(self.root / "Alto.cvc", decoder=b"retrained")  # a rebuild of the same checkpoint
        second = voice_files.cached(alto).parent
        self.assertNotEqual(second, first)
        self.assertFalse(first.exists())

        renamed = alto.rename(self.root / "Alto-final.cvc")
        voice_files.cached(renamed)
        self.assertFalse(second.exists())

        bass = make_cvc(self.root / "Bass.cvc", speaker="Bass")
        renamed.unlink()
        voice_files.cached(bass)
        self.assertEqual([p.name.split("-")[0] for p in self.cache.iterdir()], ["Bass"])


class BuildInstallTest(unittest.TestCase):
    def test_a_build_puts_the_same_file_in_the_voices_folder(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            made = make_cvc(root / "run" / "Alto-1000s.cvc")
            voices = root / "models"

            installed = voice_files.install(made, voices / made.name)
            make_cvc(made, decoder=b"rebuilt")
            voice_files.install(made, installed)  # a rebuild replaces it

            self.assertEqual([p.name for p in voices.iterdir()], ["Alto-1000s.cvc"])
            self.assertEqual(installed.read_bytes(), made.read_bytes())


    def test_builds_go_into_the_voices_folder_unless_an_older_voice_has_that_name(self):
        with tempfile.TemporaryDirectory() as temp:
            models = Path(temp)
            self.assertEqual(voice_files.voice_folder(models, "Kira"), models / "Kira")
            (models / "Kira").mkdir()
            self.assertEqual(voice_files.voice_folder(models, "Kira"), models / "Kira")
            (models / "Kira" / "Kira.causalvc.json").write_text("{}")
            self.assertEqual(voice_files.voice_folder(models, "Kira"), models)


class ExportTest(ModelFolders):
    def setUp(self):
        super().setUp()
        self.alto = checkpoint(self.root / "alto-decoder.pt", ["Alto"]).read_bytes()
        (self.root / "alto-decoder.pt").unlink()
        voice_files.write_cvc(self.root / "Alto.cvc", "Alto", "Alto", self.alto, b'{"pitch": 4.0}')
        self.out = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.out, ignore_errors=True))
        # No shared base here: the shared networks are these files.
        self.base = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.base, ignore_errors=True))
        (self.base / "encoder.pt").write_bytes(b"shared encoder")
        (self.base / "pitch.pt").write_bytes(b"shared pitch")
        patcher = mock.patch("causalvc.weights.base_folder", return_value=self.base)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_export_writes_one_file_with_the_settings_and_adds_back(self):
        engine = self.engine(selected="Alto")
        written = engine.export_model("Alto", str(self.out / "Shared Alto"))

        self.assertEqual(written, str(self.out / "Shared Alto.cvc"))
        with zipfile.ZipFile(written) as archive:
            self.assertEqual(sorted(archive.namelist()), ["decoder.pt", "realtime_settings.json", "voice.json"])
            self.assertEqual(archive.read("decoder.pt"), self.alto)
            self.assertEqual(json.loads(archive.read("realtime_settings.json"))["pitch"], 4.0)
        self.assertIn(("status", {"code": "model_exported", "name": "Alto", "path": written}), engine.events)

        with mock.patch.object(engine, "update_settings"):
            names = engine.import_model([written])
        self.assertEqual(names, ["Shared Alto"])
        self.assertEqual(decoder_in(self.root / "Shared Alto" / "Shared Alto.cvc")["speakers"], ["Alto"])

    def test_an_older_voice_folder_exports_as_one_file(self):
        folder = self.root / "Folded"
        folder.mkdir()
        checkpoint(folder / "decoder.pt", ["Folded"])
        (folder / "Folded.causalvc.json").write_text(json.dumps({"decoder": "decoder.pt", "speaker": "Folded"}))
        engine = self.engine(selected="Folded")
        engine.voice_settings["Folded"] = {"pitch": -2.0}

        written = engine.export_model("Folded", str(self.out / "Folded.cvc"))

        self.assertEqual(voice_files.read_cvc(written)["name"], "Folded")
        self.assertEqual(json.loads(voice_files.settings_in(written))["pitch"], -2.0)

    def test_export_errors(self):
        engine = self.engine()
        with self.assertRaises(core.EngineError) as raised:
            engine.export_model("Nobody", str(self.out / "x.cvc"))
        self.assertEqual(raised.exception.code, "no_model")
        with self.assertRaises(core.EngineError) as raised:
            engine.export_model("Alto", "relative.cvc")
        self.assertEqual(raised.exception.code, "bad_request")
        (self.root / "Broken").mkdir()
        (self.root / "Broken" / "Broken.causalvc.json").write_text('{"decoder": "decoder.pt", "speaker": "Broken"}')
        engine.refresh_models()
        with self.assertRaises(core.EngineError) as raised:
            engine.export_model("Broken", str(self.out / "broken.cvc"))  # its description points at no decoder
        self.assertEqual(raised.exception.code, "export_failed")

    def test_a_decoder_with_other_speakers_is_trimmed_to_this_voice(self):
        folder = self.root / "Old"
        folder.mkdir()
        checkpoint(folder / "base-decoder.pt", ["Someone", "Old", "Another"])
        (folder / "Old.causalvc.json").write_text(json.dumps({"decoder": "base-decoder.pt", "speaker": "Old"}))

        written = voice_files.folder_to_cvc(folder, self.out / "Old.cvc")

        decoder = decoder_in(written)
        self.assertEqual(decoder["speakers"], ["Old"])
        self.assertEqual(decoder["generator"]["speaker.weight"].tolist(), [[4.0, 5.0, 6.0, 7.0]])
        self.assertEqual(decoder["step"], 7)

    def test_a_voice_on_its_own_encoder_carries_it_and_the_shared_ones_stay_out(self):
        folder = self.root / "Own"
        folder.mkdir()
        (folder / "decoder.pt").write_bytes(self.alto)
        (folder / "encoder.pt").write_bytes(b"fine-tuned encoder")
        (folder / "pitch.pt").write_bytes(b"shared pitch")  # a copy of the shared one
        (folder / "Alto.causalvc.json").write_text(
            json.dumps({"decoder": "decoder.pt", "encoder": "encoder.pt", "pitch": "pitch.pt", "speaker": "Alto"})
        )

        written = voice_files.folder_to_cvc(folder, self.out / "Alto.cvc")

        with zipfile.ZipFile(written) as archive:
            self.assertNotIn("pitch.pt", archive.namelist())
            self.assertEqual(archive.read("encoder.pt"), b"fine-tuned encoder")
        installed = voice_files.unpack(written, self.out / "installed")
        self.assertEqual(json.loads(installed.read_text()), {"decoder": "decoder.pt", "speaker": "Alto", "encoder": "encoder.pt"})
        self.assertEqual((installed.parent / "encoder.pt").read_bytes(), b"fine-tuned encoder")


class ConvertScriptTest(unittest.TestCase):
    """scripts/convert_to_cvc.py: the standalone converter to hand to other people."""

    def setUp(self):
        spec = importlib.util.spec_from_file_location("convert_to_cvc", REPOSITORY / "scripts" / "convert_to_cvc.py")
        self.script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.script)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / "weights" / "causalvc-base").mkdir(parents=True)
        (self.root / "weights" / "causalvc-base" / "encoder.pt").write_bytes(b"shared encoder")
        self.models = self.root / "models"
        for name, speakers, extra in (
            ("Alto", ["Alto"], {}),
            ("Old", ["Someone", "Old"], {}),
            ("Tuned", ["Tuned"], {"encoder": "encoder.pt"}),
        ):
            folder = self.models / name
            folder.mkdir(parents=True)
            checkpoint(folder / "decoder.pt", speakers)
            (folder / f"{name}.causalvc.json").write_text(json.dumps({"decoder": "decoder.pt", "speaker": name, **extra}))
        (self.models / "Alto" / "realtime_settings.json").write_text('{"pitch": 2}')
        (self.models / "Tuned" / "encoder.pt").write_bytes(b"tuned encoder")
        (self.models / "Old" / "encoder.pt").write_bytes(b"unused")  # not named by the voice: left out

    def convert(self, *args):
        out = self.root / "out"
        with mock.patch("sys.argv", ["convert_to_cvc.py", *map(str, args), "--out", str(out)]), \
                mock.patch("builtins.print"):
            status = self.script.main()
        return status, out

    def test_a_folder_of_voices_becomes_cvc_files_the_app_reads(self):
        status, out = self.convert(self.models)

        self.assertEqual(status, 0)
        self.assertEqual(sorted(p.name for p in out.iterdir()), ["Alto.cvc", "Old.cvc", "Tuned.cvc"])
        for name in ("Alto", "Old", "Tuned"):
            self.assertEqual(voice_files.read_cvc(out / f"{name}.cvc")["speaker"], name)
            self.assertEqual(decoder_in(out / f"{name}.cvc")["speakers"], [name])
        with zipfile.ZipFile(out / "Alto.cvc") as archive:
            self.assertEqual(archive.read("decoder.pt"), (self.models / "Alto" / "decoder.pt").read_bytes())
            self.assertEqual(sorted(archive.namelist()), ["decoder.pt", "realtime_settings.json", "voice.json"])
        with zipfile.ZipFile(out / "Old.cvc") as archive:
            self.assertNotIn("encoder.pt", archive.namelist())
        installed = voice_files.unpack(out / "Tuned.cvc", self.root / "installed")
        self.assertEqual((installed.parent / "encoder.pt").read_bytes(), b"tuned encoder")
        self.assertEqual(json.loads(installed.read_text())["encoder"], "encoder.pt")

    def test_the_shared_encoder_is_left_out(self):
        (self.models / "Tuned" / "encoder.pt").write_bytes(b"shared encoder")
        status, out = self.convert(self.models / "Tuned")
        self.assertEqual(status, 0)
        with zipfile.ZipFile(out / "Tuned.cvc") as archive:
            self.assertNotIn("encoder.pt", archive.namelist())

    def test_a_voice_whose_decoder_is_missing_is_skipped(self):
        (self.models / "Alto" / "decoder.pt").unlink()
        status, out = self.convert(self.models / "Alto", self.models / "Old")
        self.assertEqual(status, 1)
        self.assertEqual([p.name for p in out.iterdir()], ["Old.cvc"])


if __name__ == "__main__":
    unittest.main()
