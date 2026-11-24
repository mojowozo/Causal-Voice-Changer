import json
import os
import tempfile
import unittest

from engine.settings import (
    EngineSettings,
    SettingError,
    coerce_setting,
    load_settings,
    load_voice_settings,
    model_settings_from,
    model_settings_to_save,
    save_settings,
    save_voice_settings,
)


class CoerceSettingTest(unittest.TestCase):
    def test_numbers_are_converted_and_clamped(self):
        self.assertEqual(coerce_setting("pitch", 12), 12.0)
        self.assertEqual(coerce_setting("block_time", 0.001), 0.01)
        self.assertEqual(coerce_setting("block_time", 0.01), 0.01)
        self.assertEqual(coerce_setting("file_input_volume", -1), 0.0)
        self.assertEqual(coerce_setting("pitch_offset_hz", 40), 40.0)
        self.assertEqual(coerce_setting("pitch_offset_hz", -5000), -300.0)
        self.assertEqual(coerce_setting("pitch_ceiling_hz", -10), 0.0)
        self.assertEqual(coerce_setting("pitch_ceiling_keep", 2), 1.0)
        self.assertEqual(coerce_setting("pitch_floor_hz", -10), 0.0)
        self.assertEqual(coerce_setting("pitch_floor_hz", 900), 500.0)
        self.assertEqual(coerce_setting("pitch_floor_keep", -0.5), 0.0)
        self.assertEqual(coerce_setting("pitch_floor_keep", 2.0), 1.0)

    def test_rejects_wrong_types_and_unknown_keys(self):
        for key, value in (
            ("pitch", "high"),
            ("pitch", True),
            ("pitch", float("nan")),
            ("input_denoise", 1),
            ("extra_time", 1.0),
            ("recording_mode", "surround"),
            ("nonsense", 1),
        ):
            with self.subTest(key=key, value=value):
                with self.assertRaises(SettingError):
                    coerce_setting(key, value)

    def test_empty_monitor_device_means_disabled(self):
        self.assertIsNone(coerce_setting("monitor_device", ""))
        self.assertIsNone(coerce_setting("monitor_device", None))
        self.assertEqual(coerce_setting("monitor_device", "[JACK] Out"), "[JACK] Out")


class PersistenceTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.directory.name, "engine.json")

    def tearDown(self):
        self.directory.cleanup()

    def test_global_settings_round_trip_without_model_settings(self):
        settings = EngineSettings(model_name="Voice", block_time=0.4, pitch=5.0)
        save_settings(settings, self.path)

        with open(self.path, encoding="utf-8") as saved_file:
            saved = json.load(saved_file)
        self.assertNotIn("pitch", saved)
        loaded = load_settings(self.path)
        self.assertEqual(loaded.model_name, "Voice")
        self.assertEqual(loaded.block_time, 0.4)
        self.assertEqual(loaded.pitch, 0.0)

    def test_invalid_saved_values_fall_back_to_defaults(self):
        with open(self.path, "w", encoding="utf-8") as config_file:
            json.dump({"block_time": "slow", "f0method": "crepe", "extra_time": 1.0, "gpu": "1"}, config_file)

        loaded = load_settings(self.path)

        self.assertEqual(loaded.block_time, EngineSettings().block_time)
        self.assertFalse(hasattr(loaded, "f0method"))  # settings of the RVC app are ignored
        self.assertEqual(loaded.gpu, "1")

    def test_old_crossfade_setting_is_ignored(self):
        # Crossfade is now a fixed 40 ms WSOLA fade (tools/splice.py).
        with open(self.path, "w", encoding="utf-8") as config_file:
            json.dump({"crossfade_time": 0.15, "block_time": 0.3}, config_file)

        loaded = load_settings(self.path)

        self.assertEqual(loaded.block_time, 0.3)
        self.assertFalse(hasattr(loaded, "crossfade_time"))
        with self.assertRaises(SettingError):
            coerce_setting("crossfade_time", 0.04)

    def test_missing_or_corrupt_file_gives_defaults(self):
        self.assertEqual(load_settings(self.path), EngineSettings())
        with open(self.path, "w", encoding="utf-8") as config_file:
            config_file.write("{not json")
        self.assertEqual(load_settings(self.path), EngineSettings())

    def test_noise_gate_is_universal_and_not_saved_per_voice(self):
        settings = EngineSettings(noise_gate_db=-40.0, pitch=3.0)
        save_settings(settings, self.path)
        loaded_global = load_settings(self.path)
        self.assertEqual(loaded_global.noise_gate_db, -40.0)

        path = os.path.join(self.directory.name, "voice_settings.json")
        save_voice_settings({"Alto": model_settings_to_save(settings)}, path)
        with open(path, encoding="utf-8") as saved_file:
            saved = json.load(saved_file)
        self.assertNotIn("noise_gate_db", saved["Alto"])
        self.assertNotIn("threhold", saved["Alto"])
        loaded = model_settings_from(load_voice_settings(path)["Alto"])
        self.assertNotIn("noise_gate_db", loaded)
        self.assertEqual(loaded["pitch"], 3.0)

    def test_damaged_voice_settings_give_defaults(self):
        path = os.path.join(self.directory.name, "voice_settings.json")
        self.assertEqual(load_voice_settings(path), {})
        with open(path, "w", encoding="utf-8") as saved_file:
            saved_file.write('{"Alto": {"pitch": "high"}, "Bass": 3')
        self.assertEqual(load_voice_settings(path), {})
        self.assertEqual(model_settings_from({"pitch": "high"})["pitch"], EngineSettings().pitch)

    def test_pitch_floor_settings_persist_and_legacy_entries_default_cleanly(self):
        legacy = model_settings_from({"pitch": 4.0})
        self.assertEqual(legacy["pitch_floor_hz"], 0.0)
        self.assertEqual(legacy["pitch_floor_keep"], 0.25)

        saved = model_settings_to_save(EngineSettings(pitch_floor_hz=185.0, pitch_floor_keep=0.2))
        self.assertEqual(saved["pitch_floor_hz"], 185.0)
        self.assertEqual(saved["pitch_floor_keep"], 0.2)


if __name__ == "__main__":
    unittest.main()
