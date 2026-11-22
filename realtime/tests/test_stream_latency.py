import queue
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from engine.core import TREBLE_SOFTENER_DELAY, RealtimeEngine, plausible_stream_latency
from engine.settings import EngineSettings
from tools.audio_fifo import AudioFrameFifo


def make_latency_engine(
    block_frame,
    native,
    in_latency=0.0,
    out_latency=0.0,
    in_stream=None,
    out_stream=None,
    fifo_in_frames=0,
    fifo_out_frames=0,
    queue_blocks=0,
    model_delay=0.050,
    denoise_delay=0.0,
    softener=False,
    file_source=False,
):
    engine = RealtimeEngine.__new__(RealtimeEngine)
    engine.samplerate = 48000
    engine.block_frame = block_frame
    engine.running = True
    engine.function = "vc"
    engine.settings = EngineSettings(
        block_time=block_frame / 48000,
        input_source="file" if file_source else "microphone",
        soften_treble=softener,
    )
    engine.file_audio_source = None
    engine.voice = SimpleNamespace(delay_seconds=model_delay)
    engine.input_stream = in_stream if in_stream is not None else (None if file_source else SimpleNamespace(latency=in_latency))
    engine.output_stream = out_stream if out_stream is not None else SimpleNamespace(latency=out_latency)
    engine.stream = engine.input_stream if (native and not file_source) else engine.output_stream
    if native and not file_source:
        engine.native_input_fifo = AudioFrameFifo(1)
        if fifo_in_frames:
            engine.native_input_fifo.write(np.zeros((fifo_in_frames, 1), dtype=np.float32))
    else:
        engine.native_input_fifo = None
    if native:
        engine.output_queue = AudioFrameFifo(1)
        if fifo_out_frames:
            engine.output_queue.write(np.zeros((fifo_out_frames, 1), dtype=np.float32))
    else:
        engine.output_queue = queue.Queue(maxsize=3)
        for _ in range(queue_blocks):
            engine.output_queue.put(np.zeros(block_frame, dtype=np.float32))
    engine.delay_time = engine.compute_initial_delay() + denoise_delay + (TREBLE_SOFTENER_DELAY if softener else 0.0)
    return engine


class StreamLatencyTest(unittest.TestCase):
    def test_keeps_a_real_latency(self):
        self.assertAlmostEqual(plausible_stream_latency(0.02), 0.02)

    def test_uses_the_output_side_of_a_duplex_pair(self):
        self.assertAlmostEqual(plausible_stream_latency((0.01, 0.03)), 0.03)

    def test_ignores_a_wrapped_jack_latency(self):
        # 2**32 - 960 frames at 48 kHz, as PortAudio reported for a PipeWire JACK node.
        self.assertEqual(plausible_stream_latency((2**32 - 960) / 48000), 0.0)

    def test_ignores_negative_and_non_finite(self):
        self.assertEqual(plausible_stream_latency(-0.01), 0.0)
        self.assertEqual(plausible_stream_latency(float("nan")), 0.0)
        self.assertEqual(plausible_stream_latency(None), 0.0)

    def test_native_jack_route_does_not_double_count_block_time(self):
        # 5.3 ms JACK in + 5.3 ms JACK out + 50 ms model + 20 ms RNNoise + 5 ms softener
        # + 44 ms in FIFOs (40 ms chunk + 4 ms infer) = 130 ms total (not 170+ ms).
        engine = make_latency_engine(
            block_frame=1920,
            native=True,
            in_latency=256 / 48000,
            out_latency=256 / 48000,
            fifo_in_frames=192,
            fifo_out_frames=1920,
            model_delay=0.050,
            denoise_delay=0.020,
            softener=True,
        )
        total_ms = round(engine.delay_time * 1000) + engine.queued_ms()
        self.assertEqual(total_ms, 130)

    def test_full_duplex_asio_reads_input_and_output_halves_once(self):
        duplex = SimpleNamespace(latency=(0.004, 0.006))
        engine = make_latency_engine(
            block_frame=960,
            native=True,
            in_stream=duplex,
            out_stream=duplex,
            fifo_in_frames=96,
            fifo_out_frames=960,
            model_delay=0.050,
        )
        self.assertAlmostEqual(engine.delay_time, 0.004 + 0.006 + 0.050, places=6)
        self.assertEqual(round(engine.delay_time * 1000) + engine.queued_ms(), 82)

    def test_block_callback_route_does_not_triple_count_block_time(self):
        # PortAudio blocksize=1920 (40 ms) reports in.latency=0.040, out.latency=0.040,
        # and 1 block in transit in queue.Queue: total with 50 ms model is 90 ms, not 170 ms.
        engine = make_latency_engine(
            block_frame=1920,
            native=False,
            in_latency=0.040,
            out_latency=0.040,
            queue_blocks=1,
            model_delay=0.050,
        )
        self.assertAlmostEqual(engine.delay_time, 0.090, places=6)
        self.assertEqual(engine.queued_ms(), 0)
        # A second block stuck in queue.Queue IS real backlog (+40 ms):
        engine.output_queue.put(np.zeros(1920, dtype=np.float32))
        self.assertEqual(engine.queued_ms(), 40)

    def test_linux_launch_scripts_request_10ms_pipewire_quantum(self):
        repo = Path(__file__).resolve().parents[2]
        for rel in ("run-realtime.sh", "realtime/run-realtime.sh"):
            text = (repo / rel).read_text(encoding="utf-8")
            self.assertIn('PIPEWIRE_QUANTUM="${PIPEWIRE_QUANTUM:-480/48000}"', text, rel)

    def test_native_inflight_inference_block_keeps_queued_ms_steady(self):
        import threading

        engine = make_latency_engine(
            block_frame=960,
            native=True,
            in_latency=256 / 48000,
            out_latency=256 / 48000,
            fifo_in_frames=960 + 96,
            fifo_out_frames=0,
            model_delay=0.050,
        )
        seen_during_callback = []
        engine.native_worker_stop = threading.Event()
        engine.native_worker_wakeup = threading.Event()
        engine.native_worker_wakeup.set()

        def fake_callback(block, frames, times, status):
            seen_during_callback.append(engine.queued_ms())
            engine.native_worker_stop.set()

        engine.audio_callback = fake_callback
        engine.native_inference_worker()
        self.assertEqual(seen_during_callback, [22])
        self.assertEqual(engine.queued_ms(), 2)

    def test_hybrid_wasapi_input_to_asio_fifo_output_latency(self):
        engine = RealtimeEngine.__new__(RealtimeEngine)
        engine.samplerate = 48000
        engine.block_frame = 1920
        engine.running = True
        engine.function = "vc"
        engine.settings = EngineSettings(block_time=0.04, input_source="microphone")
        engine.file_audio_source = None
        engine.voice = SimpleNamespace(delay_seconds=0.050)
        engine.input_stream = SimpleNamespace(latency=0.040)
        engine.output_stream = SimpleNamespace(latency=0.005)
        engine.stream = engine.output_stream
        engine.native_input_fifo = None
        engine.output_queue = AudioFrameFifo(1)
        engine.output_queue.write(np.zeros((1920, 1), dtype=np.float32))
        self.assertAlmostEqual(engine.compute_initial_delay(), 0.040 + 0.005 + 0.050, places=6)
        self.assertEqual(engine.queued_ms(), 0)
        engine.output_queue.write(np.zeros((960, 1), dtype=np.float32))
        self.assertEqual(engine.queued_ms(), 20)


if __name__ == "__main__":
    unittest.main()
