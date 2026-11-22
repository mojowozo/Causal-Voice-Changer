"""The advanced pitch settings: output pitch = input * 2^(semitones/12) + offset, then a ceiling
above which only part of the rise is kept; voiced frames only."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the causalvc package

from causalvc.fast import FastStreamingVC, map_f0_tensor  # noqa: E402
from causalvc.stream import F0_RANGE, map_f0  # noqa: E402


class MapF0Test(unittest.TestCase):
    def test_without_an_offset_it_is_the_plain_shift(self):
        f0 = torch.tensor([0.0, 100.0, 220.0])
        self.assertTrue(torch.equal(map_f0(f0, 2.0), f0 * 2.0))

    def test_offset_is_added_to_voiced_frames_only(self):
        f0 = torch.tensor([0.0, 100.0, 0.0, 200.0])
        self.assertEqual(map_f0(f0, 1.5, 40.0).tolist(), [0.0, 190.0, 0.0, 340.0])

    def test_result_stays_in_the_decoders_range(self):
        f0 = torch.tensor([0.0, 80.0, 900.0])
        self.assertEqual(map_f0(f0, 1.0, -200.0).tolist(), [0.0, F0_RANGE[0], 700.0])
        self.assertEqual(map_f0(f0, 1.0, 600.0)[2].item(), F0_RANGE[1])


    def test_ceiling_keeps_part_of_the_rise_in_semitones(self):
        f0 = torch.tensor([0.0, 120.0, 400.0])
        # +9 semitones: 120 -> 202 Hz (under the ceiling, unchanged); 400 -> 673 Hz, 14 semitones over
        # a 300 Hz ceiling, of which a quarter (3.5) is kept: about 367 Hz.
        out = map_f0(f0, 2 ** (9 / 12), ceiling=300.0, keep=0.25)
        self.assertEqual(out[0].item(), 0.0)
        self.assertAlmostEqual(out[1].item(), 120 * 2 ** (9 / 12), places=3)
        self.assertAlmostEqual(out[2].item(), 300 * (400 * 2 ** (9 / 12) / 300) ** 0.25, places=2)

    def test_keep_zero_stops_at_the_ceiling_and_one_is_no_ceiling(self):
        f0 = torch.tensor([150.0, 500.0])
        self.assertEqual(map_f0(f0, 1.0, ceiling=300.0, keep=0.0).tolist(), [150.0, 300.0])
        self.assertTrue(torch.allclose(map_f0(f0, 1.0, ceiling=300.0, keep=1.0), f0))

    def test_floor_keeps_part_of_the_drop_in_semitones_on_voiced_frames_only(self):
        f0 = torch.tensor([0.0, 135.0, 220.0])
        out = map_f0(f0, 1.0, floor=180.0, floor_keep=0.25)
        self.assertEqual(out[0].item(), 0.0)
        self.assertAlmostEqual(out[1].item(), 180.0 * (135.0 / 180.0) ** 0.25, places=3)
        self.assertAlmostEqual(out[2].item(), 220.0, places=3)

    def test_floor_keep_zero_stops_at_floor_and_one_is_no_floor(self):
        f0 = torch.tensor([0.0, 120.0, 240.0])
        self.assertEqual(map_f0(f0, 1.0, floor=180.0, floor_keep=0.0).tolist(), [0.0, 180.0, 240.0])
        self.assertTrue(torch.allclose(map_f0(f0, 1.0, floor=180.0, floor_keep=1.0), f0))

    def test_floor_above_ceiling_clamps_to_ceiling_and_matches_tensor_form(self):
        f0 = torch.tensor([0.0, 90.0, 180.0, 260.0, 420.0])
        for floor, ceiling in ((180.0, 300.0), (320.0, 250.0)):
            want = map_f0(f0, 1.25, offset=10.0, ceiling=ceiling, keep=0.3, floor=floor, floor_keep=0.2)
            p = torch.tensor([1.25, 10.0, ceiling, 0.3, floor, 0.2, 1.0])
            got = map_f0_tensor(f0, p)
            self.assertTrue(torch.allclose(got, want, atol=1e-4), f"mismatch for floor={floor}, ceiling={ceiling}: {got} vs {want}")
            # Voiced outputs must stay non-decreasing across ascending inputs even when floor > ceiling.
            voiced = got[1:]
            self.assertTrue(bool((voiced[1:] >= voiced[:-1] - 1e-5).all()))

    def test_fast_streaming_pmap_updates_floor_in_place_without_clearing_graph(self):
        vc = FastStreamingVC.__new__(FastStreamingVC)
        vc.speakers = ["Alto"]
        vc.spk = torch.zeros(1, dtype=torch.long)
        vc.pmap = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        sentinel_graph = object()
        vc.graph = sentinel_graph
        ptr = vc.pmap.data_ptr()
        vc.set_voice("Alto", semitones=11.0, offset_hz=5.0, ceiling_hz=320.0, ceiling_keep=0.35, floor_hz=185.0, floor_keep=0.2)
        self.assertEqual(vc.pmap.data_ptr(), ptr)
        self.assertIs(vc.graph, sentinel_graph)
        self.assertEqual(vc.pmap.shape, (7,))
        f0 = torch.tensor([0.0, 75.0, 120.0, 220.0])
        self.assertTrue(
            torch.allclose(
                map_f0_tensor(f0, vc.pmap),
                map_f0(f0, 2.0 ** (11.0 / 12.0), 5.0, 320.0, 0.35, 185.0, 0.2),
                atol=1e-4,
            )
        )


if __name__ == "__main__":
    unittest.main()
