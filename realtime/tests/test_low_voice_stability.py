"""Low-pitched input stability AND fast-speech syllable crispness."""

import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # causalvc package

from causalvc.decoder import model as dec_model  # noqa: E402
from causalvc.decoder.model import FRAME, SR, Decoder, loudness  # noqa: E402
from causalvc import stream  # noqa: E402


class LoudnessStabilityAndClarityTest(unittest.TestCase):
    def test_loudness_preserves_zero_lag_consonant_transients_and_220hz_calibration(self):
        t = torch.arange(SR) / SR
        tone220 = math.sqrt(2.0) * (10 ** (-20.0 / 20.0)) * torch.sin(2 * math.pi * 220.0 * t)
        db_220 = loudness(tone220[None])[0, 4:].median().item()
        self.assertAlmostEqual(db_220, -20.0, delta=0.15)

        burst = torch.zeros(1, 20 * FRAME)
        burst[0, 10 * FRAME :] = tone220[: 10 * FRAME]
        self.assertAlmostEqual(loudness(burst)[0, 10].item(), -20.0, delta=0.5)

        release = burst.flip(-1)
        self.assertAlmostEqual(loudness(release)[0, 10].item(), -100.0, delta=0.2)

        self.assertAlmostEqual(loudness(torch.zeros(1, 4 * FRAME))[0, -1].item(), -100.0, delta=0.1)
        self.assertAlmostEqual(loudness(torch.full((1, 4 * FRAME), 0.25))[0, -1].item(), -100.0, delta=0.1)
        rumble30 = math.sqrt(2.0) * (10 ** (-18.0 / 20.0)) * torch.sin(2 * math.pi * 30.0 * t)
        self.assertLess(loudness(rumble30[None])[0, 4:].median().item(), -24.5)

    def test_pitch_sync_loudness_cancels_76hz_strobe_and_82hz_ripple_while_keeping_20ms_flaps_crisp(self):
        self.assertTrue(hasattr(dec_model, "pitch_sync_loudness"), "pitch_sync_loudness is not implemented yet")
        t = torch.arange(SR) / SR

        # 1) 82 Hz harmonic train: steady within < 0.15 dB.
        low82 = sum(torch.sin(2 * math.pi * 82.0 * k * t) / k for k in range(1, 14))
        low82 = low82 / low82.pow(2).mean().sqrt() * (10 ** (-20.0 / 20.0))
        db82 = dec_model.pitch_sync_loudness(low82[None], torch.full((1, 100), 82.0))[0]
        self.assertLess((db82[8:] - db82[7:-1]).abs().max().item(), 0.15)

        # 2) 76 Hz narrow-duty-cycle glottal pulse train (>11 dB raw 10ms strobe): steady within < 0.15 dB!
        phase = (76.0 * t) % 1.0
        glottal76 = torch.exp(-phase * 5.5) * torch.sin(2 * math.pi * (8.0 * 76.0) * t)
        glottal76 = glottal76 / glottal76.pow(2).mean().sqrt() * (10 ** (-22.0 / 20.0))
        raw76 = loudness(glottal76[None])[0]
        self.assertGreater((raw76[8:] - raw76[7:-1]).abs().max().item(), 6.0)
        ps76 = dec_model.pitch_sync_loudness(glottal76[None], torch.full((1, 100), 76.0))[0]
        self.assertLess((ps76[8:] - ps76[7:-1]).abs().max().item(), 0.15)

        # 3) Fast-speech 20 ms voiced consonant flap at 95 Hz (-20 dB vowel -> -28 dB 20 ms flap -> -20 dB vowel):
        # must preserve full >=7 dB consonant dip AND recover to -20 dB with ZERO multi-frame IIR lag!
        tone95 = sum(torch.sin(2 * math.pi * 95.0 * k * t[: 20 * FRAME]) / k for k in range(1, 10))
        tone95 = tone95 / tone95.pow(2).mean().sqrt() * (10 ** (-20.0 / 20.0))
        gain = torch.ones(20 * FRAME)
        gain[8 * FRAME : 10 * FRAME] = 10 ** (-8.0 / 20.0)  # 20 ms voiced flap (frames 8 and 9)
        flap_wav = (tone95 * gain)[None]
        f95 = torch.full((1, 20), 95.0)
        db_flap = dec_model.pitch_sync_loudness(flap_wav, f95)[0]
        self.assertAlmostEqual(db_flap[7].item(), -20.0, delta=0.25)
        self.assertLess(db_flap[9].item(), -27.2)  # full consonant dip preserved!
        self.assertAlmostEqual(db_flap[11].item(), -20.0, delta=0.25)  # immediate vowel recovery, no 80 ms lag!


class SmoothF0AndToneTest(unittest.TestCase):
    def test_smooth_f0_damps_low_voice_jitter_without_lagging_glides_or_higher_pitches(self):
        jitter = torch.tensor([95.0 * (2.0 ** ((14.0 if i % 2 == 0 else -14.0) / 1200.0)) for i in range(20)])
        smoothed, _ = stream.smooth_f0(jitter)
        raw_step = 1200.0 * (jitter[1:] / jitter[:-1]).log2().abs().median().item()
        sm_step = 1200.0 * (smoothed[6:] / smoothed[5:-1]).log2().abs().median().item()
        self.assertGreater(raw_step, 25.0)
        self.assertLess(sm_step, 14.0)

        glide = torch.tensor([95.0 * (2.0 ** (35.0 * i / 1200.0)) for i in range(12)])
        sm_glide, _ = stream.smooth_f0(glide)
        glide_err = 1200.0 * (sm_glide / glide).log2().abs().max().item()
        self.assertLess(glide_err, 1.0)

        hi_voice = torch.tensor([200.0 * (2.0 ** ((14.0 if i % 2 == 0 else -14.0) / 1200.0)) for i in range(10)])
        sm_hi, _ = stream.smooth_f0(hi_voice)
        self.assertTrue(torch.allclose(sm_hi, hi_voice, atol=1e-4))

        st = torch.zeros(1)
        parts = []
        for i in range(0, len(jitter), 2):
            part, st = stream.smooth_f0(jitter[i : i + 2], st)
            parts.append(part)
        self.assertTrue(torch.allclose(torch.cat(parts), smoothed, atol=1e-5))

    def test_normalize_tone_lifts_dark_chest_mic_and_tone_gate_keeps_quiet_speech_steady(self):
        self.assertAlmostEqual(Decoder.tone_gate(torch.tensor(-48.0)).item(), 1.0, places=4)
        self.assertAlmostEqual(Decoder.tone_gate(torch.tensor(-75.0)).item(), 0.0, places=4)

        tn = torch.tensor([[-23.0, -12.0]] * 80 + [[6.0, 3.0]])
        loud = torch.full((81,), -30.0)
        f0 = torch.tensor([95.0] * 80 + [0.0])
        normed, _ = stream.normalize_tone(tn, loud, f0)
        self.assertGreater(normed[79, 0].item(), -17.5)
        self.assertGreater(normed[79, 1].item(), -8.5)
        self.assertGreater(normed[80, 0].item() - normed[79, 0].item(), 25.0)

    def test_sibilant_devoicing_is_absent_and_tail_bounds_are_asserted(self):
        # Voiced fricatives (/z/, /v/) carry their fundamental below 250 Hz while 2-8 kHz exceeds +6 dB;
        # stream/fast/infer must not strip their voicing based on tone brightness.
        self.assertFalse(hasattr(stream, "suppress_sibilant_voicing"))

        from causalvc import fast

        self.assertTrue(hasattr(fast, "check_tail_capacity"), "fast.check_tail_capacity is not implemented yet")
        fast.check_tail_capacity(5)
        fast.check_tail_capacity(6)
        with self.assertRaises(AssertionError):
            fast.check_tail_capacity(7)


if __name__ == "__main__":
    unittest.main()
