import unittest

import numpy as np

from tests.test_engine_pipeline import passthrough_engine
from tools.noise_gate import NoiseGate

RATE = 48000


class NoiseGateTest(unittest.TestCase):
    def test_gate_disabled_at_minus_60_passes_bit_exact(self):
        gate = NoiseGate(RATE)
        x = (0.002 * np.random.default_rng(0).standard_normal(1920)).astype(np.float32)
        y = gate.process(x, -60.0)
        np.testing.assert_array_equal(y, x)

    def test_quiet_floor_attenuates_and_speech_opens_without_click(self):
        gate = NoiseGate(RATE)
        t = np.arange(RATE) / RATE
        tone = (0.1 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)  # -20 dBFS
        hiss = (1e-4 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)  # -80 dBFS
        sig = np.concatenate([hiss[:4800], tone[:9600], hiss[:9600]])
        out = np.concatenate([gate.process(sig[i : i + 480], -40.0) for i in range(0, len(sig), 480)])

        self.assertLess(float(np.max(np.abs(out[2400:4800]))), 1e-6)
        np.testing.assert_allclose(out[6000:14000], sig[6000:14000], atol=1e-4)
        self.assertLess(float(np.max(np.abs(out[-1200:]))), 1e-6)
        self.assertLess(float(np.max(np.abs(np.diff(out)))), 0.01)

    def test_hold_and_hysteresis_bridge_brief_dip(self):
        gate = NoiseGate(RATE)
        t = np.arange(RATE // 2) / RATE
        loud = (0.1 * np.sin(2 * np.pi * 180 * t[:4800])).astype(np.float32)
        dip = (0.008 * np.sin(2 * np.pi * 180 * t[:960])).astype(np.float32)
        sig = np.concatenate([loud, dip, loud])
        out = gate.process(sig, -40.0)
        np.testing.assert_allclose(out[4800:5760], dip, rtol=0.05, atol=1e-4)

    def test_10ms_and_40ms_blocks_match(self):
        rng = np.random.default_rng(4)
        env = np.concatenate([np.full(4800, 1e-4), np.full(9600, 0.08), np.full(9600, 1e-4)]).astype(np.float32)
        sig = (env * rng.standard_normal(len(env))).astype(np.float32)
        g10, g40 = NoiseGate(RATE), NoiseGate(RATE)
        y10 = np.concatenate([g10.process(sig[i : i + 480], -38.0) for i in range(0, len(sig), 480)])
        y40 = np.concatenate([g40.process(sig[i : i + 1920], -38.0) for i in range(0, len(sig), 1920)])
        np.testing.assert_allclose(y10, y40, atol=1e-5)

    def test_turning_gate_off_while_closed_restores_signal_cleanly(self):
        gate = NoiseGate(RATE)
        hiss = np.full(4800, 1e-4, dtype=np.float32)
        gate.process(hiss, -35.0)  # closes gate
        x = np.full(480, 0.01, dtype=np.float32)
        y = gate.process(x, -60.0)
        np.testing.assert_array_equal(y, x)

    def test_engine_ingest_uses_smooth_noise_gate(self):
        engine = passthrough_engine(block_time=0.02)
        engine.settings.noise_gate_db = -35.0
        hiss = np.full((engine.block_frame, 1), 1e-4, dtype=np.float32)
        for _ in range(6):
            out = engine.audio_callback(hiss.copy(), engine.block_frame, None, None)
        self.assertLess(float(np.max(np.abs(out))), 1e-6)


if __name__ == "__main__":
    unittest.main()
