"""Merging voices (causalvc.merge) and the engine's merge_models command."""

import io
import json
import zipfile
from pathlib import Path
from unittest import mock

import torch

from causalvc import merge as voice_merge
from engine import core
from tests.test_model_library import ModelFolders
from tools.model_import import voice_files


def decoder_bytes(value, cfg=None, keys=("speaker.weight", "body")):
    weights = {key: torch.full((2, 3), float(value)) for key in keys}
    weights["counter"] = torch.tensor([7])  # not a float: taken from the first voice as it is
    buffer = io.BytesIO()
    torch.save({"generator": weights, "cfg": cfg or {"dim": 3}, "speakers": ["x"], "step": 5}, buffer)
    return buffer.getvalue()


def decoder_in(cvc):
    with zipfile.ZipFile(cvc) as archive:
        return torch.load(io.BytesIO(archive.read("decoder.pt")), map_location="cpu")


class MergeTest(ModelFolders):
    def setUp(self):
        super().setUp()
        (self.root / "Kira").mkdir()
        voice_files.write_cvc(self.root / "Kira" / "Kira-1000s.cvc", "Kira-1000s", "Kira", decoder_bytes(0),
                              json.dumps({"pitch": 3.0}).encode())
        voice_files.write_cvc(self.root / "Mika.cvc", "Mika", "Mika", decoder_bytes(10))
        voice_files.write_cvc(self.root / "Odd.cvc", "Odd", "Odd", decoder_bytes(5, cfg={"dim": 3, "tone": 2}))

    def test_families_group_voices_that_can_be_merged(self):
        kira, mika, odd = (voice_merge.family(self.root / p) for p in ("Kira/Kira-1000s.cvc", "Mika.cvc", "Odd.cvc"))
        self.assertEqual(kira, mika)
        self.assertNotEqual(kira, odd)
        self.assertIsNone(voice_merge.family(self.root / "Alto.cvc"))  # not a real checkpoint
        engine = self.engine()
        families = {m.name: engine.merge_family(m) for m in engine.models}
        self.assertEqual(families["Kira-1000s"], families["Mika"])

    def test_a_merge_is_the_weighted_average_written_as_a_new_voice(self):
        engine = self.engine()
        with mock.patch.object(engine, "update_settings") as select:
            name = engine.merge_models([{"name": "Kira-1000s", "strength": 75}, {"name": "Mika", "strength": 25}],
                                       "Kimi")
        self.assertEqual(name, "Kimi")
        written = self.root / "Kimi" / "Kimi.cvc"
        decoder = decoder_in(written)
        self.assertTrue(torch.allclose(decoder["generator"]["body"], torch.full((2, 3), 2.5)))
        self.assertEqual(decoder["generator"]["counter"].tolist(), [7])
        self.assertEqual(decoder["speakers"], ["Kimi"])
        self.assertEqual(decoder["merged"], [{"name": "Kira-1000s", "share": 0.75}, {"name": "Mika", "share": 0.25}])
        self.assertEqual(voice_files.read_cvc(written)["speaker"], "Kimi")
        self.assertEqual(json.loads(voice_files.settings_in(written))["pitch"], 3.0)  # the first voice's
        self.assertIn("Kimi", engine.models_by_name)
        select.assert_called_with({"model_name": "Kimi"})
        self.assertEqual([p.name for p in (self.root / "Kimi").iterdir()], ["Kimi.cvc"])

    def test_merges_that_cannot_work_are_refused(self):
        engine = self.engine()
        cases = (
            ([{"name": "Kira-1000s", "strength": 50}, {"name": "Odd", "strength": 50}], "New", "merge_incompatible"),
            ([{"name": "Kira-1000s", "strength": 50}, {"name": "Alto", "strength": 50}], "New", "merge_incompatible"),
            ([{"name": "Kira-1000s", "strength": 50}, {"name": "Mika", "strength": 0}], "New", "merge_failed"),
            ([{"name": "Mika", "strength": 50}, {"name": "Mika", "strength": 50}], "New", "merge_failed"),
            ([{"name": "Kira-1000s", "strength": 50}, {"name": "Gone", "strength": 50}], "New", "no_model"),
            ([{"name": "Kira-1000s", "strength": 50}, {"name": "Mika", "strength": 50}], "mika", "model_name_taken"),
            ([{"name": "Kira-1000s", "strength": 50}, {"name": "Mika", "strength": 50}], "a/b", "bad_model_name"),
            ([{"name": "Kira-1000s", "strength": True}], "New", "bad_request"),
        )
        for voices, name, code in cases:
            with self.assertRaises(core.EngineError, msg=(voices, name)) as raised:
                engine.merge_models(voices, name)
            self.assertEqual(raised.exception.code, code, (voices, name))
        self.assertFalse((self.root / "New").exists())

    def test_the_command_line(self):
        out = self.root / "out" / "Both.cvc"
        voice_merge.main([str(out), f"{self.root / 'Kira' / 'Kira-1000s.cvc'}=1", f"{self.root / 'Mika.cvc'}=3"])
        self.assertTrue(torch.allclose(decoder_in(out)["generator"]["body"], torch.full((2, 3), 7.5)))
