"""Low-pitched input stability without sacrificing consonant transients, glide tracking, or vocal clarity."""

import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # causalvc package

from causalvc.decoder.model import FRAME, SR, Decoder, loudness  # noqa: E402
from causalvc import stream  # noqa: E402


class LoudnessStabilityAndClarityTest(unittest.TestCase):
    def test_loudness_preserves_zero_lag_consonant_transients_and_220hz_calibration(self):
        t = torch.arange(SR) / SR
        tone220 = math.sqrt(2.0) * (10 ** (-20.0 / 20.0)) * torch.sin(2 * math.pi * 220.0 * t)
        db_220 = loudness(tone220[None])[0, 4:].median().item()
        self.assertAlmostEqual(db_220, -20.0, delta=0.15)

        # Step onset (-100 -> -20 dB at frame 10): consonant burst must not be crushed by a lagged window.
        burst = torch.zeros(1, 20 * FRAME)
        burst[0, 10 * FRAME :] = tone220[: 10 * FRAME]
        self.assertAlmostEqual(loudness(burst)[0, 10].item(), -20.0, delta=0.5)

        # Step release (-20 -> -100 dB at frame 10): stop-consonant closure must drop immediately to silence.
        release = burst.flip(-1)
        self.assertAlmostEqual(loudness(release)[0, 10].item(), -100.0, delta=0.2)

        # Digital silence and pure DC offset both read -100 dB; 30 Hz desk rumble is attenuated.
        self.assertAlmostEqual(loudness(torch.zeros(1, 4 * FRAME))[0, -1].item(), -100.0, delta=0.1)
        self.assertAlmostEqual(loudness(torch.full((1, 4 * FRAME), 0.25))[0, -1].item(), -100.0, delta=0.1)
        rumble30 = math.sqrt(2.0) * (10 ** (-18.0 / 20.0)) * torch.sin(2 * math.pi * 30.0 * t)
        self.assertLess(loudness(rumble30[None])[0, 4:].median().item(), -24.5)

    def test_smooth_loud_damps_low_f0_glottal_ripple_without_blurring_transients_or_high_f0(self):
        self.assertTrue(hasattr(stream, "smooth_loud"), "stream.smooth_loud is not implemented yet")
        t = torch.arange(SR) / SR
        low_voice = sum(torch.sin(2 * math.pi * 82.0 * k * t) / k for k in range(1, 14))
        low_voice = low_voice / low_voice.pow(2).mean().sqrt() * (10 ** (-20.0 / 20.0))
        raw_db = loudness(low_voice[None])[0]
        sm_db, _ = stream.smooth_loud(raw_db, torch.full_like(raw_db, 82.0))
        step_jitter = (sm_db[8:] - sm_db[7:-1]).abs().max().item()
        self.assertLess(step_jitter, 0.42)

        # Sharp consonant onset and stop closure pass with zero lag even at 85 Hz.
        env = torch.tensor([-100.0] * 5 + [-20.0] * 5 + [-100.0] * 5)
        f85 = torch.tensor([0.0] * 5 + [85.0] * 5 + [0.0] * 5)
        sm_env, _ = stream.smooth_loud(env, f85)
        self.assertAlmostEqual(sm_env[5].item(), -20.0, places=3)
        self.assertAlmostEqual(sm_env[10].item(), -100.0, places=3)

        # Voices >= 140 Hz pass untouched, and streaming state matches whole-sequence output.
        sm_hi, _ = stream.smooth_loud(raw_db, torch.full_like(raw_db, 200.0))
        self.assertTrue(torch.allclose(sm_hi, raw_db, atol=1e-6))

        st = None
        parts = []
        f82 = torch.full_like(raw_db, 82.0)
        for i in range(0, len(raw_db), 2):
            part, st = stream.smooth_loud(raw_db[i : i + 2], f82[i : i + 2], st)
            parts.append(part)
        self.assertTrue(torch.allclose(torch.cat(parts), sm_db, atol=1e-5))


class SmoothF0AndToneTest(unittest.TestCase):
    def test_smooth_f0_damps_low_voice_jitter_without_lagging_glides_or_higher_pitches(self):
        # 20 frames of 95 Hz with alternating +-14 cent 10 ms micro-jitter.
        jitter = torch.tensor([95.0 * (2.0 ** ((14.0 if i % 2 == 0 else -14.0) / 1200.0)) for i in range(20)])
        smoothed, _ = stream.smooth_f0(jitter)
        raw_step = 1200.0 * (jitter[1:] / jitter[:-1]).log2().abs().median().item()
        sm_step = 1200.0 * (smoothed[6:] / smoothed[5:-1]).log2().abs().median().item()
        self.assertGreater(raw_step, 25.0)
        self.assertLess(sm_step, 14.0)

        # Real intonation glide (35 cents / 10 ms from 95 Hz) and >=150 Hz voices do not lag.
        glide = torch.tensor([95.0 * (2.0 ** (35.0 * i / 1200.0)) for i in range(12)])
        sm_glide, _ = stream.smooth_f0(glide)
        glide_err = 1200.0 * (sm_glide / glide).log2().abs().max().item()
        self.assertLess(glide_err, 1.0)

        hi_voice = torch.tensor([200.0 * (2.0 ** ((14.0 if i % 2 == 0 else -14.0) / 1200.0)) for i in range(10)])
        sm_hi, _ = stream.smooth_f0(hi_voice)
        self.assertTrue(torch.allclose(sm_hi, hi_voice, atol=1e-4))

        # Block-by-block state carry matches whole-sequence output.
        st = torch.zeros(1)
        parts = []
        for i in range(0, len(jitter), 2):
            part, st = stream.smooth_f0(jitter[i : i + 2], st)
            parts.append(part)
        self.assertTrue(torch.allclose(torch.cat(parts), smoothed, atol=1e-5))

    def test_normalize_tone_lifts_dark_chest_mic_and_tone_gate_keeps_quiet_speech_steady(self):
        self.assertTrue(hasattr(stream, "normalize_tone"), "stream.normalize_tone is not implemented yet")
        # Quiet speech at -48 dBFS must keep full tone conditioning (not get pulled toward [0, 0] dB),
        # while dead air at -75 dBFS is gated to 0.
        self.assertAlmostEqual(Decoder.tone_gate(torch.tensor(-48.0)).item(), 1.0, places=4)
        self.assertAlmostEqual(Decoder.tone_gate(torch.tensor(-75.0)).item(), 0.0, places=4)

        # Dark male close-mic vowels at [-23, -12] dB get lifted toward studio speech presence,
        # while a sharp unvoiced sibilant burst (+6 dB) keeps its upward jump.
        tn = torch.tensor([[-23.0, -12.0]] * 80 + [[6.0, 3.0]])
        loud = torch.full((81,), -30.0)
        f0 = torch.tensor([95.0] * 80 + [0.0])
        normed, _ = stream.normalize_tone(tn, loud, f0)
        self.assertGreater(normed[79, 0].item(), -17.5)
        self.assertGreater(normed[79, 1].item(), -8.5)
        self.assertGreater(normed[80, 0].item() - normed[79, 0].item(), 25.0)


if __name__ == "__main__":
    unittest.main()
