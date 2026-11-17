"""A voice's index (causalvc.index): carried in its .cvc, kept through copies and merges,
reported to the UI, and blended in at the rate its Index slider sets."""

import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import torch

from causalvc import index as voice_index
from causalvc import merge as voice_merge
from engine.settings import MODEL_SETTING_KEYS, coerce_setting
from tests.test_voice_merge import decoder_bytes
from tools.model_import import voice_files


def index_bytes(frames):
    buffer = io.BytesIO()
    torch.save({"frames": frames}, buffer)
    return buffer.getvalue()


def frames_in(cvc):
    with zipfile.ZipFile(cvc) as archive:
        return torch.load(io.BytesIO(archive.read("index.pt")), map_location="cpu")["frames"]


class IndexMathTest(unittest.TestCase):
    def test_representative_frames_stand_for_clusters(self):
        torch.manual_seed(0)
        centres = torch.nn.functional.normalize(torch.randn(4, 16), dim=-1) * 5
        frames = (centres.repeat_interleave(50, 0) + 0.05 * torch.randn(200, 16)).half()
        picked = voice_index.representative(frames, size=4).float()
        self.assertEqual(picked.shape, (4, 16))
        nearest = torch.cdist(centres, picked).min(1).values
        self.assertLess(float(nearest.max()), 0.2)

    def test_a_frame_in_the_index_looks_up_itself(self):
        torch.manual_seed(1)
        bank = torch.randn(64, 32)
        found = voice_index.lookup(bank[:5], bank, torch.nn.functional.normalize(bank, dim=-1))
        self.assertTrue(torch.allclose(found, bank[:5], atol=1e-3))


class IndexFileTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_an_index_added_to_a_voice_is_unpacked_and_found(self):
        cvc = voice_files.write_cvc(self.root / "Alto.cvc", "Alto", "Alto", decoder_bytes(1))
        self.assertIsNone(voice_files.load(cvc)["index"])
        voice_files.replace_member(cvc, voice_files.CVC_INDEX, index_bytes(torch.ones(3, 4)))
        self.assertEqual(voice_files.read_cvc(cvc)["speaker"], "Alto")
        found = voice_files.load(cvc)["index"]
        self.assertIsNotNone(found)
        self.assertTrue(torch.equal(torch.load(found)["frames"], torch.ones(3, 4)))
        # Saving settings into the file keeps the index.
        copy = voice_files.copy_with_settings(cvc, self.root / "Copy.cvc", json.dumps({"pitch": 2}).encode())
        self.assertTrue(torch.equal(frames_in(copy), torch.ones(3, 4)))

    def test_an_index_does_not_stop_voices_merging_and_goes_along(self):
        a = voice_files.write_cvc(self.root / "A.cvc", "A", "A", decoder_bytes(0), index=index_bytes(torch.zeros(2, 4)))
        b = voice_files.write_cvc(self.root / "B.cvc", "B", "B", decoder_bytes(4))
        c = voice_files.write_cvc(self.root / "C.cvc", "C", "C", decoder_bytes(8), index=index_bytes(torch.ones(3, 4)))
        self.assertEqual(voice_merge.family(a), voice_merge.family(b))
        self.assertEqual(voice_merge.family(a), voice_merge.family(c))
        one = voice_merge.merge([(a, 1), (b, 1)], self.root / "AB.cvc", "AB")
        self.assertEqual(frames_in(one).shape, (2, 4))
        both = voice_merge.merge([(a, 1), (c, 1)], self.root / "AC.cvc", "AC")
        self.assertEqual(frames_in(both).shape, (3, 4))  # as many as the larger index
        none = voice_merge.merge([(b, 1), (b, 1)], self.root / "BB.cvc", "BB")
        with zipfile.ZipFile(none) as archive:
            self.assertNotIn("index.pt", archive.namelist())

    def test_a_merged_index_follows_the_strengths(self):
        a = voice_files.write_cvc(self.root / "A.cvc", "A", "A", decoder_bytes(0), index=index_bytes(torch.zeros(100, 4)))
        a2 = voice_files.write_cvc(self.root / "A2.cvc", "A2", "A", decoder_bytes(1), index=index_bytes(torch.zeros(100, 4)))
        c = voice_files.write_cvc(self.root / "C.cvc", "C", "C", decoder_bytes(8), index=index_bytes(torch.ones(100, 4)))
        merged = frames_in(voice_merge.merge([(a, 90), (c, 10)], self.root / "AC.cvc", "AC"))
        self.assertEqual(len(merged), 100)
        self.assertEqual(int((merged[:, 0] == 1).sum()), 10)
        # Two checkpoints of one voice share its index: their strengths add up (60 + 20 against 20).
        merged = frames_in(voice_merge.merge([(a, 60), (a2, 20), (c, 20)], self.root / "AAC.cvc", "AAC"))
        self.assertEqual(int((merged[:, 0] == 0).sum()), 80)

    def test_the_rate_is_a_voice_setting_from_0_to_1(self):
        self.assertIn("index_rate", MODEL_SETTING_KEYS)
        self.assertEqual(coerce_setting("index_rate", 1.5), 1.0)
        self.assertEqual(coerce_setting("index_rate", -1), 0.0)


if __name__ == "__main__":
    unittest.main()
