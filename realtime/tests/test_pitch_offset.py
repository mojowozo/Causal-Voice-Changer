"""The advanced pitch settings: output pitch = input * 2^(semitones/12) + offset, then a ceiling
above which only part of the rise is kept; voiced frames only."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the causalvc package

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
        self.assertEqual(map_f0(f0, 1.0, 300.0)[2].item(), F0_RANGE[1])


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


if __name__ == "__main__":
    unittest.main()
