"""Bandwidth detection (detect_fmax) and colour/warp-invariant tone conditioning in decoder training."""

import tempfile
import unittest
from pathlib import Path

import soundfile as sf
import torch

from causalvc.decoder import train


class DetectFmaxAndToneAugTest(unittest.TestCase):
    def test_detect_fmax_caps_44_1khz_and_17khz_livestream_cutoffs_but_not_full_48khz(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            torch.manual_seed(0)

            # 1) 44.1 kHz recording with broadband energy up to its 22.05 kHz Nyquist -> capped around ~20.9 kHz.
            x44 = 0.08 * torch.randn(44100 * 4)
            p44 = root / "voice44k.wav"
            sf.write(str(p44), x44.numpy(), 44100)
            fmax_44 = train.detect_fmax([str(p44)])
            self.assertIsNotNone(fmax_44)
            self.assertGreater(fmax_44, 19500.0)
            self.assertLess(fmax_44, 21500.0)

            # 2) 48 kHz speech-tilted recording with an AAC/stream MDCT cutoff at 16 kHz.
            x48 = torch.randn(48000 * 4)
            freq = torch.fft.rfftfreq(len(x48), 1 / 48000)
            speech_tilt = (freq.clamp_min(100.0) / 1000.0) ** (-0.8)
            lp = speech_tilt * torch.sigmoid((16000.0 - freq) / 120.0)
            x_aac = torch.fft.irfft(torch.fft.rfft(x48) * lp, len(x48))
            x_aac = 0.1 * x_aac / x_aac.pow(2).mean().sqrt()
            p_aac = root / "stream_aac.wav"
            sf.write(str(p_aac), x_aac.numpy(), 48000, subtype="FLOAT")
            fmax_aac = train.detect_fmax([str(p_aac)])
            self.assertIsNotNone(fmax_aac)
            self.assertLessEqual(fmax_aac, 18500.0)

            # 3) True full-band 48 kHz audio with gentle natural spectral rolloff -> None (full 24 kHz kept).
            natural = torch.fft.irfft(torch.fft.rfft(x48) * (freq.clamp_min(100.0) / 1000.0) ** (-0.6), len(x48))
            natural = 0.1 * natural / natural.pow(2).mean().sqrt()
            p_full = root / "studio48k.wav"
            sf.write(str(p_full), natural.numpy(), 48000)
            self.assertIsNone(train.detect_fmax([str(p_full)]))

    def test_training_tone_targets_follow_spectral_warp_not_mic_colour(self):
        self.assertTrue(hasattr(train, "training_tone"), "train.training_tone is not implemented yet")
        torch.manual_seed(1)
        t = torch.arange(48000) / 48000.0
        # Harmonic signal with strong 600 Hz energy vs weaker 2500 Hz energy.
        clean = (0.2 * torch.sin(2 * 3.14159265 * 600.0 * t) + 0.03 * torch.sin(2 * 3.14159265 * 2500.0 * t))[None]
        # Muffled copy (simulating mic colour low-pass removing 2500 Hz): training_tone must read `clean`, not `muffled`,
        # and when `warp_k` stretches spectrum upward (k < 20 -> higher frequencies in resampled frame), tone shifts accordingly.
        tn_clean = train.training_tone(clean, spread=0.0)
        tn_warped = train.training_tone(clean, spread=0.0, warp_k=14)
        self.assertEqual(tn_clean.shape, (1, 100, 2))
        self.assertEqual(tn_warped.shape, (1, 100, 2))
        self.assertGreater((tn_warped - tn_clean).abs().mean().item(), 1.0)


if __name__ == "__main__":
    unittest.main()
