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


class CheckpointVoiceTest(ModelFolders):
    """models/Kira/ with three checkpoints, beside the loose Alto, Bass and Tenor."""

    def setUp(self):
        super().setUp()
        (self.root / "Kira").mkdir()
        for name in ("Kira-1000s", "Kira-2000s", "Kira-10000s"):
            voice_files.write_cvc(self.root / "Kira" / f"{name}.cvc", name, "Kira", b"decoder")

    def voice(self, engine, name):
        (voice,) = [voice for voice in engine.state_voices() if voice["name"] == name]
        return voice

    def test_checkpoints_group_into_one_voice_by_step(self):
        voice_files.write_cvc(self.root / "Solo-500s.cvc", "Solo-500s", "Solo", b"decoder")
        voice_files.write_cvc(self.root / "Solo-1500s.cvc", "Solo-1500s", "Solo", b"decoder")
        groups = model_registry.group_voices(discover_models(self.root))
        self.assertEqual([group.name for group in groups], ["Alto", "Bass", "Kira", "Solo", "Tenor"])
        kira = groups[2]
        self.assertEqual([entry.name for entry in kira.checkpoints], ["Kira-1000s", "Kira-2000s", "Kira-10000s"])
        self.assertEqual([entry.step for entry in kira.checkpoints], [1000, 2000, 10000])
        self.assertEqual(kira.latest.name, "Kira-10000s")
        self.assertEqual([entry.name for entry in groups[3].checkpoints], ["Solo-500s", "Solo-1500s"])
        self.assertEqual(groups[0].latest.name, "Alto")
        self.assertIsNone(groups[0].checkpoints[0].step)

    def test_a_voices_folder_groups_any_name_and_steps_need_a_separator(self):
        voice_files.write_cvc(self.root / "Kira" / "Kira-best.cvc", "Kira-best", "Kira", b"decoder")
        (kira,) = [g for g in model_registry.group_voices(discover_models(self.root)) if g.name == "Kira"]
        self.assertEqual(kira.checkpoints[-1].name, "Kira-best")  # no step: after the steps
        self.assertEqual(kira.latest.name, "Kira-10000s")
        self.assertEqual(model_registry.checkpoint_step("Kye_40e_7600s"), 7600)
        self.assertIsNone(model_registry.checkpoint_step("Mika2s"))

    def test_state_lists_each_voice_once(self):
        engine = self.engine(selected="Bass")
        state = core.RealtimeEngine.state_voices(engine)
        self.assertEqual([voice["name"] for voice in state], ["Alto", "Bass", "Kira", "Tenor"])
        kira = self.voice(engine, "Kira")
        self.assertEqual(kira["checkpoints"], ["Kira-1000s", "Kira-2000s", "Kira-10000s"])
        self.assertEqual(kira["checkpoint"], "Kira-10000s")

    def test_checkpoints_share_the_voices_settings(self):
        engine = self.engine(selected="Kira-10000s")
        engine.update_settings({"pitch": 5.0, "output_gain_db": 3.0})
        engine.update_settings({"model_name": "Kira-2000s"})
        self.assertEqual((engine.settings.pitch, engine.settings.output_gain_db), (5.0, 3.0))
        self.assertEqual(self.saved_voice_settings()["Kira"]["pitch"], 5.0)
        self.assertNotIn("Kira-10000s", self.saved_voice_settings())
        engine.update_settings({"model_name": "Bass"})
        self.assertEqual(engine.settings.pitch, 0.0)
        engine.update_settings({"model_name": "Kira-1000s"})
        self.assertEqual(engine.settings.pitch, 5.0)

    def test_the_picked_checkpoint_is_remembered_until_the_newest_is_picked(self):
        engine = self.engine(selected="Bass")
        engine.update_settings({"model_name": "Kira-2000s"})
        self.assertEqual(self.voice(engine, "Kira")["checkpoint"], "Kira-2000s")
        self.assertEqual(self.saved_voice_settings()["Kira"]["checkpoint"], "Kira-2000s")
        engine.update_settings({"model_name": "Bass"})
        self.assertEqual(self.voice(engine, "Kira")["checkpoint"], "Kira-2000s")
        engine.update_settings({"model_name": "Kira-10000s"})
        self.assertEqual(self.voice(engine, "Kira")["checkpoint"], "Kira-10000s")
        self.assertNotIn("Kira", self.saved_voice_settings())  # nothing else saved for it yet
        # A newer checkpoint arriving later is then the one picked.
        voice_files.write_cvc(self.root / "Kira" / "Kira-20000s.cvc", "Kira-20000s", "Kira", b"decoder")
        engine.reload_models()
        self.assertEqual(self.voice(engine, "Kira")["checkpoint"], "Kira-20000s")

    def test_settings_saved_per_checkpoint_move_to_the_voice(self):
        engine = self.engine(selected="Kira-2000s")
        engine.voice_settings = {
            "Kira-1000s": {"pitch": 1.0},
            "Kira-2000s": {"pitch": 2.0, "threhold": -40.0},
            "Kira-10000s": {"pitch": 10.0},
            "Bass": {"pitch": -2.0},
        }
        engine.migrate_voice_settings()
        saved = self.saved_voice_settings()
        # The one in use last wins, and is remembered as the picked checkpoint; nothing is dropped.
        self.assertEqual(saved["Kira"], {"pitch": 2.0, "threhold": -40.0, "checkpoint": "Kira-2000s"})
        self.assertEqual(saved["Kira-1000s"], {"pitch": 1.0})
        self.assertEqual(saved["Bass"], {"pitch": -2.0})
        self.assertEqual(engine.model_settings_for("Kira-10000s")["pitch"], 2.0)
        self.assertEqual(engine.model_settings_for("Kira-1000s")["noise_gate_db"], -40.0)
        # Run again: the voice's own entry now wins.
        engine.voice_settings["Kira-10000s"] = {"pitch": 12.0}
        self.voice_settings_file.clear()
        engine.migrate_voice_settings()
        self.assertEqual(self.saved_voice_settings(), {})

    def test_without_the_selection_the_newest_checkpoints_settings_move(self):
        engine = self.engine(selected="Bass")
        engine.voice_settings = {"Kira-1000s": {"pitch": 1.0}, "Kira-2000s": {"pitch": 2.0}}
        engine.migrate_voice_settings()
        self.assertEqual(self.saved_voice_settings()["Kira"], {"pitch": 2.0})

    def test_settings_a_checkpoint_carries_are_the_voices(self):
        voice_files.write_cvc(self.root / "Kira" / "Kira-10000s.cvc", "Kira-10000s", "Kira", b"decoder",
                              json.dumps({"pitch": 8.0}).encode())
        engine = self.engine()
        self.assertEqual(engine.model_settings_for("Kira-1000s")["pitch"], 8.0)

    def test_deleting_a_checkpoint_keeps_the_voice_and_its_settings(self):
        engine = self.engine(selected="Kira-10000s")
        engine.update_settings({"pitch": 3.0})
        engine.flush_model_settings_save(force=True)
        with mock.patch.object(core, "move_to_trash", side_effect=self.trash):
            engine.delete_model("Kira-10000s")
        self.assertEqual(engine.settings.model_name, "Kira-2000s")  # the voice stays selected
        self.assertEqual(engine.settings.pitch, 3.0)
        self.assertEqual(self.saved_voice_settings()["Kira"]["pitch"], 3.0)
        with mock.patch.object(core, "move_to_trash", side_effect=self.trash):
            engine.delete_model("Kira-2000s")
            engine.delete_model("Kira-1000s")
        self.assertNotIn("Kira", engine.voice_settings)

    def test_deleting_or_renaming_the_picked_checkpoint(self):
        engine = self.engine(selected="Kira-1000s")
        engine.remember_checkpoint("Kira-1000s")
        engine.update_settings({"model_name": "Bass"})
        engine.rename_model("Kira-1000s", "Kira-early")
        self.assertEqual(self.voice(engine, "Kira")["checkpoint"], "Kira-early")
        with mock.patch.object(core, "move_to_trash", side_effect=self.trash):
            engine.delete_model("Kira-early")
        self.assertEqual(self.voice(engine, "Kira")["checkpoint"], "Kira-10000s")

    def test_switching_checkpoints_while_converting_swaps_only_the_voice(self):
        engine = self.engine(selected="Kira-10000s")
        engine.settings.pitch = 4.0
        engine.config = SimpleNamespace(device="cpu")
        old = mock.Mock(delay_seconds=0.06)
        engine.voice, engine.running, engine.block_frame, engine.delay_time = old, True, 1920, 0.2
        built = mock.Mock(delay_seconds=0.08)
        engine.stop = mock.Mock()
        with mock.patch.object(core.causalvc_backend, "CausalVC", return_value=built) as make:
            engine.update_settings({"model_name": "Kira-2000s"})
        engine.stop.assert_not_called()
        self.assertIs(engine.voice, built)
        self.assertEqual(make.call_args.args[:2], (4.0, str((self.root / "Kira" / "Kira-2000s.cvc").resolve())))
        built.warm.assert_called_once_with(1920, record=False)
        built.record.assert_called_once_with(1920)
        self.assertAlmostEqual(engine.delay_time, 0.22)
        self.assertIn(("status", {"code": "checkpoint_switched", "name": "Kira-2000s"}), engine.events)
        self.assertEqual(engine.settings.pitch, 4.0)

    def test_a_checkpoint_that_fails_to_load_keeps_the_old_one(self):
        engine = self.engine(selected="Kira-10000s")
        engine.config = SimpleNamespace(device="cpu")
        old = mock.Mock(delay_seconds=0.06)
        engine.voice, engine.running, engine.block_frame, engine.delay_time = old, True, 1920, 0.2
        engine.stop = mock.Mock()
        with mock.patch.object(core.causalvc_backend, "CausalVC", side_effect=RuntimeError("bad file")):
            with self.assertRaises(core.EngineError) as raised:
                engine.update_settings({"model_name": "Kira-2000s"})
        self.assertEqual(raised.exception.code, "model_load_failed")
        self.assertIs(engine.voice, old)
        self.assertEqual(engine.settings.model_name, "Kira-10000s")
        engine.stop.assert_not_called()

    def test_another_voice_while_converting_still_stops(self):
        engine = self.engine(selected="Kira-10000s")
        engine.voice, engine.running = mock.Mock(), True
        engine.stop = mock.Mock()
        with mock.patch.object(core.causalvc_backend, "CausalVC") as make:
            engine.update_settings({"model_name": "Bass"})
        engine.stop.assert_called_once()
        make.assert_not_called()

    def test_a_loose_checkpoint_renamed_out_of_its_voice_takes_the_settings(self):
        voice_files.write_cvc(self.root / "Solo-500s.cvc", "Solo-500s", "Solo", b"decoder")
        voice_files.write_cvc(self.root / "Solo-1500s.cvc", "Solo-1500s", "Solo", b"decoder")
        engine = self.engine(selected="Solo-1500s")
        engine.voice_settings = {"Solo": {"pitch": 6.0}}
        engine.rename_model("Solo-1500s", "Duet")
        self.assertEqual(engine.model_settings_for("Duet")["pitch"], 6.0)
        self.assertEqual(engine.model_settings_for("Solo-500s")["pitch"], 6.0)


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
