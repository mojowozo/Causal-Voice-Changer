import tempfile
import unittest
from pathlib import Path

from tools.model_registry import discover_models


class DiscoverModelsTest(unittest.TestCase):
    def test_discovers_direct_model_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            alpha = root / "Alpha"
            alpha.mkdir()
            (alpha / "voice.causalvc.json").touch()

            empty = root / "Empty"
            empty.mkdir()
            (empty / "notes.txt").touch()

            entries = discover_models(root)

            self.assertEqual([entry.name for entry in entries], ["Alpha"])
            self.assertEqual(entries[0].model_path.name, "voice.causalvc.json")

    def test_earlier_file_name_still_loads(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = root / "Older"
            model_dir.mkdir()
            (model_dir / "voice.nextvc.json").touch()

            entries = discover_models(root)

            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].model_path.name, "voice.nextvc.json")


if __name__ == "__main__":
    unittest.main()
