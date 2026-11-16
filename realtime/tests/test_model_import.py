import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.model_import import ModelImportError, import_models, plan_import, voice_files
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

    def make_voice(self, name, settings=None):
        path = self.downloads / name
        voice_files.write_cvc(path, Path(name).stem, "Alto", b"decoder", settings)
        return str(path)

    def test_each_voice_file_is_copied_in_as_a_cvc(self):
        names = import_models(
            [self.make_voice("Alto.cvc"), self.make_voice("Bass.cvc"), self.make("notes.txt")], self.models
        )

        self.assertEqual(names, ["Alto", "Bass"])
        self.assertTrue((self.models / "Alto" / "Alto.cvc").is_file())
        self.assertTrue((self.models / "Bass" / "Bass.cvc").is_file())
        entries = discover_models(self.models)
        self.assertEqual([entry.name for entry in entries], ["Alto", "Bass"])
        self.assertEqual(entries[0].model_path.name, "Alto.cvc")
        self.assertTrue(entries[0].is_file)

    def test_a_name_in_use_gets_a_suffix(self):
        self.models.mkdir()
        voice_files.write_cvc(self.models / "Voice.cvc", "Voice", "Alto", b"decoder")
        (self.models / "Older").mkdir()

        names = import_models([self.make_voice("Voice.cvc"), self.make_voice("Older.cvc")], self.models)

        self.assertEqual(names, ["Voice (2)", "Older (2)"])
        self.assertTrue((self.models / "Voice" / "Voice (2).cvc").is_file())

    def test_checkpoints_go_into_their_voices_folder(self):
        self.models.mkdir()
        (self.models / "Kira").mkdir()
        voice_files.write_cvc(self.models / "Kira" / "Kira-1000s.cvc", "Kira-1000s", "Kira", b"decoder")

        names = import_models([self.make_voice("Kira-2000s.cvc"), self.make_voice("Kira-1000s.cvc")], self.models)

        self.assertEqual(names, ["Kira-2000s", "Kira-1000s (2)"])
        self.assertEqual(sorted(p.name for p in (self.models / "Kira").iterdir()),
                         ["Kira-1000s (2).cvc", "Kira-1000s.cvc", "Kira-2000s.cvc"])
        self.assertEqual([e.name for e in discover_models(self.models)], ["Kira-1000s", "Kira-1000s (2)", "Kira-2000s"])

    def test_an_older_voice_folder_of_that_name_is_left_alone(self):
        older = self.models / "Kira"
        older.mkdir(parents=True)
        (older / "decoder.pt").write_bytes(b"x")
        (older / "Kira.causalvc.json").write_text('{"decoder": "decoder.pt"}')

        import_models([self.make_voice("Kira-2000s.cvc")], self.models)

        self.assertTrue((self.models / "Kira-2000s.cvc").is_file())
        self.assertEqual(sorted(e.name for e in discover_models(self.models)), ["Kira", "Kira-2000s"])

    def test_a_file_that_is_not_a_voice_is_refused(self):
        with self.assertRaises(ModelImportError) as raised:
            import_models([self.make("Fake.cvc", b"not a zip")], self.models)

        self.assertEqual(raised.exception.code, "import_failed")
        self.assertEqual(list(self.models.iterdir()), [])

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
        files = [self.make_voice("Voice.cvc")]

        def copy_then_fail(source, destination):
            Path(destination).write_bytes(b"half")
            raise OSError("No space left on device")

        with mock.patch("tools.model_import.shutil.copyfile", side_effect=copy_then_fail):
            with self.assertRaises(ModelImportError) as raised:
                import_models(files, self.models)

        self.assertIn("No space left", raised.exception.message)
        self.assertEqual(list(self.models.iterdir()), [])


class HiddenFolderTest(unittest.TestCase):
    def test_hidden_entries_are_not_models(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            staging = Path(temp_dir) / ".import-abc123"
            staging.mkdir()
            (staging / "Voice.causalvc.json").touch()
            voice_files.write_cvc(Path(temp_dir) / ".import-def456.cvc", "Voice", "Alto", b"decoder")

            self.assertEqual(discover_models(temp_dir), [])


if __name__ == "__main__":
    unittest.main()
