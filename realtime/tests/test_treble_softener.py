import unittest

import numpy as np
from scipy.signal import butter, sosfiltfilt

from tools.treble_softener import DELAY_SECONDS, FRAME, SAMPLE_RATE, TrebleSoftener

DELAY = round(DELAY_SECONDS * SAMPLE_RATE)


def voice(seconds, treble_db, start=0.0):
    """A 200 Hz voice: harmonics to 3 kHz at a steady level, and harmonics from 4 to 10 kHz
    whose level (dB, per sample) follows ``treble_db``; silent before ``start`` seconds."""
    t = np.arange(round(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    body = sum(np.sin(2 * np.pi * 200 * k * t) / k for k in range(1, 16))
    treble = sum(np.sin(2 * np.pi * 200 * k * t) for k in range(20, 51)) / 31
    audio = 0.05 * body + 0.05 * treble * 10 ** (treble_db(t) / 20)
    audio[t < start] = 0.0
    return audio.astype(np.float32)


def stream(audio, block=960):
    softener = TrebleSoftener()
    return np.concatenate([softener.process(audio[i:i + block]) for i in range(0, len(audio), block)])


def fastest_treble_rise(audio, start=0.45, end=0.6):
    """The biggest 5 ms rise of the 4-12 kHz band's 5 ms average between ``start`` and ``end`` seconds, in dB."""
    band = sosfiltfilt(butter(4, [4000, 12000], "bandpass", fs=SAMPLE_RATE, output="sos"), audio) ** 2
    window = SAMPLE_RATE // 200
    level = 10 * np.log10(np.convolve(band, np.ones(window) / window, "same")[round(start * SAMPLE_RATE):round(end * SAMPLE_RATE):FRAME] + 1e-12)
    return float((level[5:] - level[:-5]).max())


class TrebleSoftenerTest(unittest.TestCase):
    def test_steady_voice_passes_unchanged_only_delayed(self):
        audio = voice(0.6, lambda t: np.full_like(t, -6.0))
        out = stream(audio)
        np.testing.assert_allclose(out[DELAY:], audio[:-DELAY], atol=1e-6)
        np.testing.assert_array_equal(out[:DELAY], 0.0)

    def test_sudden_treble_mid_word_is_stretched(self):
        # Treble 16 dB louder within 1 ms, half a second into a sounding voice.
        audio = voice(1.0, lambda t: -22.0 + np.clip((t - 0.5) / 0.001, 0, 1) * 16)
        out = stream(audio)
        self.assertGreater(fastest_treble_rise(audio), 10.0)
        self.assertLess(fastest_treble_rise(out[DELAY:]), fastest_treble_rise(audio) - 5.0)
        # Afterwards the treble is back to its full level.
        np.testing.assert_allclose(out[DELAY + SAMPLE_RATE * 9 // 10:], audio[SAMPLE_RATE * 9 // 10:-DELAY], atol=1e-6)

    def test_a_word_starting_from_silence_is_untouched(self):
        audio = voice(0.6, lambda t: np.full_like(t, -6.0), start=0.3)
        out = stream(audio)
        np.testing.assert_allclose(out[DELAY:], audio[:-DELAY], atol=1e-6)

    def test_a_word_after_a_long_noisy_pause_is_untouched(self):
        # Ten seconds of quiet background noise: it must not come to count as a voice.
        noise = (np.random.default_rng(0).standard_normal(SAMPLE_RATE * 10) * 3e-4).astype(np.float32)
        audio = np.concatenate([noise, voice(0.6, lambda t: np.full_like(t, -6.0)) + noise[: SAMPLE_RATE * 6 // 10]])
        out = stream(audio)
        np.testing.assert_allclose(out[DELAY:], audio[:-DELAY], atol=1e-6)

    def test_block_size_does_not_change_the_result(self):
        audio = voice(1.0, lambda t: -22.0 + np.clip((t - 0.5) / 0.001, 0, 1) * 16)
        np.testing.assert_allclose(stream(audio, 960), stream(audio, 1920), atol=1e-7)
        np.testing.assert_allclose(stream(audio, 960), stream(audio, FRAME), atol=1e-7)

    def test_blocks_must_be_whole_frames(self):
        with self.assertRaises(ValueError):
            TrebleSoftener().process(np.zeros(FRAME + 1, dtype=np.float32))

    def test_linear_phase_split_has_no_midrange_cancellation_notch_at_minus_6db(self):
        from scipy.signal import freqz
        from tools.treble_softener import SPLIT_HALF

        softener = TrebleSoftener()
        w, H_lo = freqz(softener.split_fir, worN=4096, fs=SAMPLE_RATE)
        A = np.real(H_lo * np.exp(1j * 2 * np.pi * w / SAMPLE_RATE * SPLIT_HALF))
        # At -6 dB treble attenuation, the old Butterworth IIR split had a -40.7 dB notch at 2.6 kHz;
        # the linear-phase FIR split must keep 100-3000 Hz flat within +-0.05 dB.
        g = 10 ** (-6.0 / 20.0)
        db = 20 * np.log10(np.abs(A + g * (1.0 - A)) + 1e-12)
        mid = (w >= 100) & (w <= 3000)
        self.assertLess(float(np.abs(db[mid]).max()), 0.05)


if __name__ == "__main__":
    unittest.main()
