import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from engine import core
from engine.settings import EngineSettings
from tools import model_registry
from tools.model_import import voice_files
from tools.model_registry import (
    TrashError,
    discover_models,
    model_name_problem,
    move_to_trash,
    rename_model,
)


class MoveToTrashTest(unittest.TestCase):
    def test_uses_gio_trash(self):
        done = subprocess.CompletedProcess([], 0, "", "")
        with mock.patch.object(model_registry.shutil, "which", return_value="/usr/bin/gio"), \
             mock.patch.object(model_registry.subprocess, "run", return_value=done) as run:
            move_to_trash(Path("/models/Voice"))
        self.assertEqual(run.call_args.args[0], ["/usr/bin/gio", "trash", "--", "/models/Voice"])

    def test_reports_failures_and_a_missing_gio(self):
        failed = subprocess.CompletedProcess([], 1, "", "Unable to find or create trash directory")
        with mock.patch.object(model_registry.shutil, "which", return_value="/usr/bin/gio"), \
             mock.patch.object(model_registry.subprocess, "run", return_value=failed):
            with self.assertRaisesRegex(TrashError, "trash directory"):
                move_to_trash(Path("/models/Voice"))
        with mock.patch.object(model_registry.shutil, "which", return_value=None):
            with self.assertRaisesRegex(TrashError, "gio"):
                move_to_trash(Path("/models/Voice"))


class ModelFolders(unittest.TestCase):
    """models/ with Alto.cvc (carrying pitch 4), Bass.cvc and Tenor.cvc, and an engine without devices."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for name in ("Alto", "Bass", "Tenor"):
            settings = json.dumps({"pitch": 4.0}).encode() if name == "Alto" else None
            voice_files.write_cvc(self.root / f"{name}.cvc", name, name, b"decoder", settings)
        patcher = mock.patch.object(core, "MODELS_ROOT", str(self.root))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.temp.cleanup)
        save = mock.patch.object(core, "save_settings")
        self.saved = save.start()
        self.addCleanup(save.stop)
        # configs/voice_settings.json, in memory.
        self.voice_settings_file = {}
        store = mock.patch.object(
            core, "save_voice_settings",
            side_effect=lambda values: self.voice_settings_file.update({"saved": json.loads(json.dumps(values))}),
        )
        store.start()
        self.addCleanup(store.stop)

    def saved_voice_settings(self):
        return self.voice_settings_file.get("saved", {})

    def engine(self, selected="Bass"):
        engine = core.RealtimeEngine.__new__(core.RealtimeEngine)
        engine.settings = EngineSettings(model_name=selected)
        engine.devices = SimpleNamespace(
            resolve=lambda value, kind: value,
            resolve_monitor=lambda value: value,
            inputs={},
            outputs={},
        )
        engine.gpu_options = [{"id": "auto", "label": None}]
        engine.pending_model_settings_name = None
        engine.model_settings_save_due = 0.0
        engine.voice_settings = {}
        engine.running = False
        engine.function = "vc"
        engine.events = []
        engine.emit = lambda event, data: engine.events.append((event, data))
        engine.state = lambda: {}
        engine.refresh_models()
        return engine

    def trash(self, path):
        # Stand-in for gio: the voice leaves models/ (hidden entries are not models).
        os.rename(path, self.root / f".trashed-{Path(path).name}")


class DeleteModelTest(ModelFolders):
    def test_deleting_another_model_keeps_the_selection(self):
        engine = self.engine(selected="Bass")
        with mock.patch.object(core, "move_to_trash", side_effect=self.trash) as trash:
            engine.delete_model("Tenor")
        self.assertEqual(trash.call_args.args[0], (self.root / "Tenor.cvc").resolve())
        self.assertEqual([model.name for model in engine.models], ["Alto", "Bass"])
        self.assertEqual(engine.settings.model_name, "Bass")
        self.assertIn(("status", {"code": "model_deleted", "name": "Tenor"}), engine.events)

    def test_deleting_the_selected_model_selects_another_with_its_settings(self):
        engine = self.engine(selected="Bass")
        with mock.patch.object(core, "move_to_trash", side_effect=self.trash):
            engine.delete_model("Bass")
        self.assertEqual(engine.settings.model_name, "Alto")
        self.assertEqual(engine.settings.pitch, 4.0)
        self.saved.assert_called()

    def test_a_deleted_voice_leaves_no_settings_behind(self):
        engine = self.engine(selected="Bass")
        engine.settings.pitch = 7.0
        engine.schedule_model_settings_save()
        with mock.patch.object(core, "move_to_trash", side_effect=self.trash):
            engine.delete_model("Bass")
        self.assertFalse((self.root / "Bass.cvc").exists())
        self.assertNotIn("Bass", engine.voice_settings)
        self.assertNotIn("Bass", self.saved_voice_settings())

    def test_the_converting_model_cannot_be_deleted(self):
        engine = self.engine(selected="Bass")
        engine.running = True
        with mock.patch.object(core, "move_to_trash") as trash:
            with self.assertRaises(core.EngineError) as raised:
                engine.delete_model("Bass")
        self.assertEqual(raised.exception.code, "model_in_use")
        trash.assert_not_called()

    def test_failures_and_unknown_models_are_reported(self):
        engine = self.engine()
        with self.assertRaises(core.EngineError) as raised:
            engine.delete_model("Nobody")
        self.assertEqual(raised.exception.code, "no_model")
        with mock.patch.object(core, "move_to_trash", side_effect=TrashError("no trash")):
            with self.assertRaises(core.EngineError) as raised:
                engine.delete_model("Alto")
        self.assertEqual(raised.exception.code, "delete_failed")
        self.assertTrue((self.root / "Alto.cvc").exists())


class RenameModelTest(ModelFolders):

    def test_renaming_keeps_settings_and_the_selection(self):
        engine = self.engine(selected="Alto")
        engine.rename_model("Alto", "Alto-final_200e_14200s")
        self.assertEqual([m.name for m in engine.models], ["Alto-final_200e_14200s", "Bass", "Tenor"])
        self.assertEqual(engine.settings.model_name, "Alto-final_200e_14200s")
        self.assertEqual(engine.settings.pitch, 4.0)
        self.assertTrue((self.root / "Alto-final_200e_14200s.cvc").is_file())
        self.assertFalse((self.root / "Alto.cvc").exists())
        self.saved.assert_called()
        self.assertIn(
            ("status", {"code": "model_renamed", "name": "Alto", "new_name": "Alto-final_200e_14200s"}),
            engine.events,
        )

    def test_pending_settings_follow_the_new_name(self):
        engine = self.engine(selected="Bass")
        engine.settings.pitch = 9.5
        engine.schedule_model_settings_save()
        engine.rename_model("Bass", "Baritone")
        self.assertFalse((self.root / "Bass.cvc").exists())
        self.assertEqual(self.saved_voice_settings()["Baritone"]["pitch"], 9.5)
        self.assertNotIn("Bass", self.saved_voice_settings())
        self.assertEqual(engine.model_settings_for("Baritone")["pitch"], 9.5)

    def test_case_only_rename(self):
        engine = self.engine()
        engine.rename_model("Tenor", "TENOR")
        self.assertEqual([m.name for m in engine.models], ["Alto", "Bass", "TENOR"])

    def test_rejected_names(self):
        engine = self.engine()
        cases = {
            "Bass": "model_name_taken",
            "bass": "model_name_taken",
            "": "bad_model_name",
            " Alto2": "bad_model_name",
            ".hidden": "bad_model_name",
            "a/b": "bad_model_name",
            "what?": "bad_model_name",
        }
        for new_name, code in cases.items():
            with self.subTest(new_name=new_name):
                with self.assertRaises(core.EngineError) as raised:
                    engine.rename_model("Alto", new_name)
                self.assertEqual(raised.exception.code, code)
        self.assertTrue((self.root / "Alto.cvc").is_file())

    def test_the_converting_model_cannot_be_renamed(self):
        engine = self.engine(selected="Bass")
        engine.running = True
        with self.assertRaises(core.EngineError) as raised:
            engine.rename_model("Bass", "Baritone")
        self.assertEqual(raised.exception.code, "model_in_use")
        engine.rename_model("Tenor", "Countertenor")  # other models are fine

    def test_name_rules(self):
        self.assertIsNone(model_name_problem("Sakura-shortstream-50m-1.6_200e_14200s"))
        self.assertIsNone(model_name_problem("さくら 声"))
        self.assertEqual(model_name_problem("trailing."), "edges")
        self.assertEqual(model_name_problem("x" * 201), "too_long")
        self.assertEqual(model_name_problem("tab\there"), "characters")

    def test_rename_refuses_to_overwrite(self):
        with self.assertRaises(FileExistsError):
            rename_model(self.root / "Alto.cvc", "Bass")
        self.assertTrue((self.root / "Alto.cvc").is_file())


class VoiceSettingsTest(ModelFolders):
    def test_settings_are_kept_by_name_not_in_the_voice(self):
        engine = self.engine(selected="Bass")
        before = (self.root / "Bass.cvc").read_bytes()
        engine.settings.pitch = -3.0
        engine.schedule_model_settings_save()
        engine.flush_model_settings_save(force=True)
        self.assertEqual(self.saved_voice_settings()["Bass"]["pitch"], -3.0)
        self.assertEqual((self.root / "Bass.cvc").read_bytes(), before)

    def test_saved_settings_win_over_the_ones_a_voice_carries(self):
        engine = self.engine()
        self.assertEqual(engine.model_settings_for("Alto")["pitch"], 4.0)
        engine.voice_settings["Alto"] = {"pitch": -1.0}
        self.assertEqual(engine.model_settings_for("Alto")["pitch"], -1.0)

    def test_export_packs_the_current_settings(self):
        engine = self.engine()
        engine.voice_settings["Alto"] = {"pitch": 2.5}
        written = engine.export_model("Alto", str(self.root / "out" / "Shared"))
        self.assertTrue(written.endswith("Shared.cvc"))
        self.assertEqual(json.loads(voice_files.settings_in(written))["pitch"], 2.5)
        self.assertEqual(voice_files.read_cvc(written)["speaker"], "Alto")


class OlderVoiceFolderTest(ModelFolders):
    def setUp(self):
        super().setUp()
        (self.root / "Older").mkdir()
        (self.root / "Older" / "Older.causalvc.json").write_text('{"decoder": "decoder.pt", "speaker": "Older"}')
        (self.root / "Older" / "realtime_settings.json").write_text('{"pitch": 6.0}')

    def test_older_folders_still_list_with_their_settings(self):
        engine = self.engine(selected="Older")
        self.assertEqual([m.name for m in engine.models], ["Alto", "Bass", "Older", "Tenor"])
        self.assertFalse(engine.models_by_name["Older"].is_file)
        self.assertEqual(engine.model_settings_for("Older")["pitch"], 6.0)

    def test_a_cvc_wins_over_a_folder_of_the_same_name(self):
        voice_files.write_cvc(self.root / "Older.cvc", "Older", "Older", b"decoder")
        (entry,) = [m for m in discover_models(self.root) if m.name == "Older"]
        self.assertTrue(entry.is_file)

    def test_older_folders_rename_as_folders(self):
        engine = self.engine()
        engine.rename_model("Older", "Oldest")
        self.assertTrue((self.root / "Oldest" / "Older.causalvc.json").is_file())


class VoiceFolderTest(ModelFolders):
    """models/Kira/ holding two checkpoints, beside the loose files."""

    def setUp(self):
        super().setUp()
        (self.root / "Kira").mkdir()
        for name in ("Kira-1000s", "Kira-2000s"):
            voice_files.write_cvc(self.root / "Kira" / f"{name}.cvc", name, "Kira", b"decoder")

    def test_each_checkpoint_in_a_voices_folder_is_a_voice(self):
        (self.root / "Kira" / ".import-x.cvc").write_bytes(b"partial")
        voice_files.write_cvc(self.root / "Kira-1000s.cvc", "Kira-1000s", "Kira", b"loose")  # loose files win
        entries = discover_models(self.root)
        self.assertEqual([e.name for e in entries], ["Alto", "Bass", "Kira-1000s", "Kira-2000s", "Tenor"])
        self.assertEqual(entries[2].path, (self.root / "Kira-1000s.cvc").resolve())
        self.assertEqual(entries[3].path, (self.root / "Kira" / "Kira-2000s.cvc").resolve())

    def test_deleting_the_last_checkpoint_removes_the_folder(self):
        engine = self.engine(selected="Bass")
        with mock.patch.object(core, "move_to_trash", side_effect=self.trash):
            engine.delete_model("Kira-1000s")
            self.assertTrue((self.root / "Kira").is_dir())
            engine.delete_model("Kira-2000s")
        self.assertFalse((self.root / "Kira").exists())
        self.assertEqual([model.name for model in engine.models], ["Alto", "Bass", "Tenor"])

    def test_a_rename_stays_in_the_voices_folder(self):
        engine = self.engine(selected="Bass")
        engine.rename_model("Kira-2000s", "Kira-best")
        self.assertTrue((self.root / "Kira" / "Kira-best.cvc").is_file())
        self.assertIn("Kira-best", [model.name for model in engine.models])


class RegistryMetadataTest(unittest.TestCase):
    def test_size_of_the_decoder_and_date_of_the_voice_file(self):
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / "Voice").mkdir()
            decoder = Path(temp) / "decoder.pt"
            decoder.write_bytes(b"x" * 1234)
            model = Path(temp) / "Voice" / "Voice.causalvc.json"
            model.write_text('{"decoder": "%s", "pitch": "p.pt", "speaker": "Voice"}' % decoder)
            os.utime(model, (1_700_000_000, 1_700_000_000))
            (entry,) = discover_models(temp)
        self.assertEqual((entry.size_bytes, entry.modified), (1234, 1_700_000_000))


if __name__ == "__main__":
    unittest.main()
