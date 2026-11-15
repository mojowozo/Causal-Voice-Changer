import threading
import unittest
from unittest import mock

import numpy as np

from engine import core
from tools import hushwire, rnnoise

RATE = rnnoise.SAMPLE_RATE


def voice_like(seconds, f0=140.0):
    """A harmonic signal with a speech-like spectral tilt and syllable envelope."""
    t = np.arange(int(seconds * RATE)) / RATE
    tone = sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, 20))
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t) ** 2
    return (0.1 * tone * envelope).astype(np.float32)


def level_db(audio):
    return 10 * np.log10(np.mean(np.square(audio, dtype=np.float64)) + 1e-20)


@unittest.skipUnless(rnnoise.available(), "librnnoise not installed (sudo pacman -S rnnoise)")
class RNNoiseTest(unittest.TestCase):
    def test_rejects_partial_frames_and_keeps_length(self):
        denoiser = rnnoise.RNNoise()
        with self.assertRaises(ValueError):
            denoiser.process(np.zeros(100, dtype=np.float32))
        self.assertEqual(denoiser.process(np.zeros(4800, dtype=np.float32)).shape, (4800,))

    def test_output_delay_matches_the_engine_accounting(self):
        signal = voice_like(2.0)
        denoiser = rnnoise.RNNoise()
        output = np.concatenate(
            [denoiser.process(signal[i : i + 4800]) for i in range(0, signal.shape[0], 4800)]
        )
        lags = range(0, 2000)
        scores = [float(np.dot(signal[RATE // 2 : -2000], output[RATE // 2 + lag : -2000 + lag])) for lag in lags]
        # A periodic test tone cannot pin the peak to one sample; allow ±2.
        self.assertLessEqual(abs(int(np.argmax(scores)) - int(core.RNNOISE_DELAY_SECONDS * RATE)), 2)

    def test_suppresses_noise_and_keeps_voice(self):
        rng = np.random.default_rng(0)
        noise = (rng.standard_normal(RATE * 2) * 0.02).astype(np.float32)
        voice = voice_like(2.0)
        noise_out = rnnoise.RNNoise().process(noise)
        voice_out = rnnoise.RNNoise().process(voice)
        # Skip the first half second while the network settles.
        settle = RATE // 2
        self.assertLess(level_db(noise_out[settle:]) - level_db(noise[settle:]), -10.0)
        self.assertGreater(level_db(voice_out[settle:]) - level_db(voice[settle:]), -6.0)


def denoising_engine(method):
    from tests.test_engine_pipeline import passthrough_engine

    engine = passthrough_engine(block_time=0.1)
    engine.settings.input_denoise = True
    engine.settings.input_denoise_method = method
    engine.running = True
    engine.delay_time = 0.1
    engine.refresh_denoiser()
    return engine


def run(engine, signal):
    block = engine.block_frame
    return np.concatenate(
        [
            engine.audio_callback(signal[i : i + block, None].copy(), block, None, None).copy()
            for i in range(0, signal.shape[0] - block + 1, block)
        ]
    )


class DenoiserSelectionTest(unittest.TestCase):
    def test_no_denoising_and_no_delay_without_rnnoise(self):
        with mock.patch.object(rnnoise, "available", return_value=False):
            engine = denoising_engine("rnnoise")
        self.assertIsNone(engine.active_denoiser)
        self.assertAlmostEqual(engine.delay_time, 0.1)

    def test_hushwire_without_its_model_leaves_the_input_alone(self):
        with mock.patch.object(hushwire, "model_ready", return_value=False):
            engine = denoising_engine("hushwire")
            self.assertEqual(engine.hushwire_status(), "needs_download" if hushwire.available() else "unavailable")
        self.assertIsNone(engine.active_denoiser)
        self.assertAlmostEqual(engine.delay_time, 0.1)

    def test_hushwire_is_the_default(self):
        from engine.settings import EngineSettings

        self.assertEqual(EngineSettings().input_denoise_method, "hushwire")

    @unittest.skipUnless(rnnoise.available(), "librnnoise not installed")
    def test_passthrough_with_rnnoise_delays_by_20_ms(self):
        engine = denoising_engine("rnnoise")
        self.assertIsInstance(engine.active_denoiser, rnnoise.RNNoise)
        self.assertAlmostEqual(engine.delay_time, 0.1 + core.RNNOISE_DELAY_SECONDS)
        signal = voice_like(3.0)
        output = run(engine, signal)
        lag = int(core.RNNOISE_DELAY_SECONDS * RATE)
        start = RATE
        correlation = np.corrcoef(output[start:], signal[start - lag : output.shape[0] - lag])[0, 1]
        self.assertGreater(correlation, 0.95)

    @unittest.skipUnless(hushwire.available() and hushwire.model_ready(), "Hushwire's model is not downloaded")
    def test_switching_methods_and_turning_off_keep_the_delay_in_step(self):
        engine = denoising_engine("hushwire")
        self.assertIsInstance(engine.active_denoiser, hushwire.Hushwire)
        hushwire_delay = engine.active_denoiser.delay_seconds(engine.block_frame)
        self.assertAlmostEqual(engine.delay_time, 0.1 + hushwire_delay)
        if rnnoise.available():
            engine.settings.input_denoise_method = "rnnoise"
            engine.refresh_denoiser()
            self.assertAlmostEqual(engine.delay_time, 0.1 + core.RNNOISE_DELAY_SECONDS)
        engine.settings.input_denoise = False
        engine.refresh_denoiser()
        self.assertIsNone(engine.active_denoiser)
        self.assertAlmostEqual(engine.delay_time, 0.1)


@unittest.skipUnless(hushwire.available() and hushwire.model_ready(), "Hushwire's model is not downloaded")
class HushwireTest(unittest.TestCase):
    def test_delay_matches_hushwire(self):
        # 100 ms blocks are not whole 128-frame quanta: one quantum of padding, like Hushwire's plugin.
        engine = denoising_engine("hushwire")
        self.assertEqual(round(engine.active_denoiser.delay_seconds(engine.block_frame) * RATE), 2651)
        # Whole quanta (the default 40 ms blocks) need none.
        self.assertEqual(round(hushwire.Hushwire().delay_seconds(1920) * RATE), 2523)

    def test_blocks_of_any_size_line_up(self):
        # With the network bypassed the model is a pure delay, so an impulse shows the
        # buffering exactly; the output is negated to undo the network's polarity flip.
        for block in (4800, 1920, 1000):
            denoiser = hushwire.Hushwire()
            denoiser._model.enabled = False
            signal = np.zeros(RATE, dtype=np.float32)
            signal[20000] = 1.0
            output = np.concatenate([denoiser.process(signal[i : i + block]) for i in range(0, RATE - block + 1, block)])
            peak = int(np.argmax(np.abs(output)))
            self.assertEqual(peak - 20000, denoiser.delay_frames - hushwire.MODEL_DELAY_FRAMES, block)
            self.assertAlmostEqual(float(output[peak]), -1.0, places=4)

    def test_removes_noise(self):
        noise = (np.random.default_rng(0).standard_normal(RATE * 3) * 0.02).astype(np.float32)
        noise_out = hushwire.Hushwire().process(noise)
        self.assertLess(level_db(noise_out[RATE:]) - level_db(noise[RATE:]), -20.0)


class HushwireDownloadTest(unittest.TestCase):
    def test_nothing_downloads_before_the_notice_is_accepted(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as folder, mock.patch.object(hushwire, "data_dir", return_value=Path(folder)):
            with mock.patch("urllib.request.urlopen") as urlopen:
                with self.assertRaises(hushwire.DownloadNotAccepted):
                    hushwire.ensure_model()
                urlopen.assert_not_called()
            hushwire.accept_download()
            self.assertTrue(hushwire.download_accepted())


    @unittest.skipUnless(hushwire.available() and hushwire.model_ready(), "Hushwire's model is not downloaded")
    def test_accepting_the_notice_downloads_and_starts_denoising(self):
        import shutil
        import tempfile
        from pathlib import Path

        cached = hushwire.model_path()
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(hushwire, "data_dir", return_value=Path(folder)):

            def fake_download(variant=hushwire.DEFAULT_VARIANT):
                self.assertTrue(hushwire.download_accepted())  # consent is recorded first
                hushwire.model_path().parent.mkdir(parents=True)
                shutil.copy(cached, hushwire.model_path())
                return hushwire.model_path()

            engine = denoising_engine("hushwire")
            events = []
            engine.emit = lambda event, data: events.append((event, data))
            engine.state = lambda: {}
            self.assertIsNone(engine.active_denoiser)
            self.assertEqual(engine.hushwire_status(), "needs_download")
            with mock.patch.object(hushwire, "ensure_model", side_effect=fake_download):
                engine.download_hushwire()
                self.assertEqual(engine.hushwire_status(), "downloading")
                for thread in threading.enumerate():
                    if thread.name == "hushwire-download":
                        thread.join(10)
            self.assertIn(("status", {"code": "hushwire_downloaded"}), events)
            self.assertIsInstance(engine.active_denoiser, hushwire.Hushwire)
            self.assertAlmostEqual(engine.delay_time, 0.1 + engine.active_denoise_delay)


if __name__ == "__main__":
    unittest.main()
