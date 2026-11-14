import os
import shutil
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from tools.model_import import ModelImportError, import_models, plan_import
from tools.model_registry import discover_models


class ModelImportTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.downloads = self.root / "Downloads"
        self.downloads.mkdir()
        self.models = self.root / "models"

    def tearDown(self):
        self.temp.cleanup()

    def make(self, name, content=b"weights"):
        path = self.downloads / name
        path.write_bytes(content)
        return str(path)

    def test_each_voice_file_becomes_a_folder(self):
        names = import_models(
            [self.make("Alto.causalvc.json"), self.make("Bass.nextvc.json"), self.make("notes.txt")], self.models
        )

        self.assertEqual(names, ["Alto", "Bass"])
        entries = discover_models(self.models)
        self.assertEqual([entry.name for entry in entries], ["Alto", "Bass"])
        self.assertEqual(entries[0].model_path.name, "Alto.causalvc.json")
        self.assertEqual((self.models / "Alto").stat().st_mode & 0o777, 0o755)

    def test_existing_folder_name_gets_a_suffix(self):
        (self.models / "Voice").mkdir(parents=True)

        names = import_models([self.make("Voice.causalvc.json")], self.models)

        self.assertEqual(names, ["Voice (2)"])
        self.assertTrue((self.models / "Voice (2)" / "Voice.causalvc.json").is_file())

    def test_requires_a_voice_file(self):
        with self.assertRaises(ModelImportError) as raised:
            import_models([self.make("weights.pt")], self.models)

        self.assertEqual(raised.exception.code, "import_no_model_file")
        self.assertFalse(self.models.exists())

    def test_missing_file_is_reported(self):
        with self.assertRaises(ModelImportError) as raised:
            plan_import([str(self.downloads / "gone.causalvc.json")])

        self.assertEqual(raised.exception.code, "import_failed")

    def test_failed_copy_leaves_no_partial_model(self):
        files = [self.make("Voice.causalvc.json")]

        def copy_then_fail(source, destination):
            raise OSError("No space left on device")

        with mock.patch("tools.model_import.shutil.copy2", side_effect=copy_then_fail):
            with self.assertRaises(ModelImportError) as raised:
                import_models(files, self.models)

        self.assertIn("No space left", raised.exception.message)
        self.assertEqual(list(self.models.iterdir()), [])


class HiddenFolderTest(unittest.TestCase):
    def test_hidden_folders_are_not_models(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            staging = Path(temp_dir) / ".import-abc123"
            staging.mkdir()
            (staging / "Voice.causalvc.json").touch()

            self.assertEqual(discover_models(temp_dir), [])


if __name__ == "__main__":
    unittest.main()
