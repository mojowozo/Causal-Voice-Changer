"""Low-pitched input stability: 20 ms high-passed Hann loudness and causal F0 micro-jitter smoothing."""

import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # causalvc package

from causalvc.decoder.model import FRAME, SR, loudness  # noqa: E402
from causalvc import stream  # noqa: E402


class LoudnessStabilityTest(unittest.TestCase):
    def test_low_pitched_glottal_train_has_steady_loudness_and_keeps_220hz_calibration(self):
        t = torch.arange(SR) / SR
        # 82 Hz harmonic glottal-like pulse train (12.2 ms period vs 10 ms frame clock).
        low_voice = sum(torch.sin(2 * math.pi * 82.0 * k * t) / k for k in range(1, 14))
        low_voice = low_voice / low_voice.pow(2).mean().sqrt() * (10 ** (-20.0 / 20.0))
        db_low = loudness(low_voice[None])[0, 4:]
        step_jitter = (db_low[1:] - db_low[:-1]).abs().max().item()
        self.assertLess(step_jitter, 0.35)

        # Calibration check on a 220 Hz tone at -20 dBFS RMS: existing .cvc models expect -20 dB.
        tone220 = math.sqrt(2.0) * (10 ** (-20.0 / 20.0)) * torch.sin(2 * math.pi * 220.0 * t)
        db_220 = loudness(tone220[None])[0, 4:].median().item()
        self.assertAlmostEqual(db_220, -20.0, delta=0.15)

        # Digital silence still reads -100 dB.
        self.assertAlmostEqual(loudness(torch.zeros(1, 4 * FRAME))[0, -1].item(), -100.0, delta=0.1)

    def test_sub_80hz_desk_rumble_is_attenuated_in_loudness(self):
        t = torch.arange(SR) / SR
        rumble30 = math.sqrt(2.0) * (10 ** (-18.0 / 20.0)) * torch.sin(2 * math.pi * 30.0 * t)
        db_rumble = loudness(rumble30[None])[0, 4:].median().item()
        self.assertLess(db_rumble, -30.0)


class SmoothF0Test(unittest.TestCase):
    def test_damps_10ms_micro_jitter_without_lagging_jumps_or_voicing_onsets(self):
        self.assertTrue(hasattr(stream, "smooth_f0"), "stream.smooth_f0 is not implemented yet")
        # 20 frames of 170 Hz with alternating +-18 cent 10 ms micro-jitter.
        jitter = torch.tensor([170.0 * (2.0 ** ((18.0 if i % 2 == 0 else -18.0) / 1200.0)) for i in range(20)])
        smoothed, state = stream.smooth_f0(jitter)
        raw_step = 1200.0 * (jitter[1:] / jitter[:-1]).log2().abs().median().item()
        sm_step = 1200.0 * (smoothed[6:] / smoothed[5:-1]).log2().abs().median().item()
        self.assertGreater(raw_step, 30.0)
        self.assertLess(sm_step, 14.0)

        # A large +500 cent musical step and an unvoiced -> voiced onset lock immediately without slur.
        seq = torch.tensor([0.0, 0.0, 170.0, 170.0 * (2.0 ** (500.0 / 1200.0)), 0.0, 220.0])
        whole, _ = stream.smooth_f0(seq)
        self.assertEqual(whole[0].item(), 0.0)
        self.assertAlmostEqual(whole[2].item(), 170.0, places=3)
        self.assertAlmostEqual(whole[3].item(), seq[3].item(), places=2)
        self.assertEqual(whole[4].item(), 0.0)
        self.assertAlmostEqual(whole[5].item(), 220.0, places=3)

        # Block-by-block state carry matches whole-sequence output exactly.
        st = torch.zeros(1)
        parts = []
        for i in range(0, len(jitter), 2):
            part, st = stream.smooth_f0(jitter[i : i + 2], st)
            parts.append(part)
        self.assertTrue(torch.allclose(torch.cat(parts), smoothed, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
