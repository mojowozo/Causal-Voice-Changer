"""Run the real engine buffers in passthrough mode.

Passthrough skips the model, so the output must be the input, block for
block, with no added delay.  This checks the buffer handling end to end.
"""

import threading
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from engine.core import RealtimeEngine
from engine.settings import EngineSettings
from tools.wav_recorder import WavRecorder

RATE = 48000


def passthrough_engine(block_time):
    engine = RealtimeEngine.__new__(RealtimeEngine)
    engine.settings = EngineSettings(block_time=block_time)
    engine.config = SimpleNamespace(device=torch.device("cpu"))
    engine.voice = SimpleNamespace(
        reset_stream_state=lambda: None, warm=lambda frames: None, delay_seconds=0.06
    )
    engine.function = "im"
    engine.running = False
    engine.recorder = WavRecorder()
    engine.monitor_device_index = None
    engine.output_queue = None
    engine.monitor_queue = None
    engine.last_input_meter_update = 0.0
    engine.last_output_meter_update = 0.0
    engine.emit = lambda event, data: None
    engine.denoise_lock = threading.RLock()
    engine.hushwire_download = None
    engine.prepare_inference(RATE)
    return engine


class PassthroughPipelineTest(unittest.TestCase):
    def run_signal(self, engine, signal):
        outputs = []
        block = engine.block_frame
        for start in range(0, signal.shape[0] - block + 1, block):
            chunk = signal[start : start + block, None].copy()
            outputs.append(engine.audio_callback(chunk, block, None, None).copy())
        return np.concatenate(outputs)

    def test_output_is_the_input(self):
        engine = passthrough_engine(block_time=0.1)
        rng = np.random.default_rng(3)
        signal = (rng.standard_normal(RATE * 2) * 0.1).astype(np.float32)

        output = self.run_signal(engine, signal)

        np.testing.assert_allclose(output, signal[: output.shape[0]], atol=1e-5)

    def test_block_length_is_a_whole_number_of_20_ms_frames(self):
        for block_time, frames in ((0.02, 960), (0.04, 1920), (0.06, 2880), (0.1, 4800)):
            with self.subTest(block_time=block_time):
                self.assertEqual(passthrough_engine(block_time=block_time).block_frame, frames)


if __name__ == "__main__":
    unittest.main()
