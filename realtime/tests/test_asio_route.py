"""ASIO routes: PortAudio opens one stream per ASIO driver, so input, output and
monitor on the same ASIO device must share one stream (a second open fails with
"Device unavailable [PaErrorCode -9985]")."""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
from test_engine_pipeline import passthrough_engine

from engine.core import EngineError
from engine.devices import Endpoint
from engine.settings import SettingError, coerce_setting
from tools.asio import BufferSizes
from tools.audio_routing import ASIO_API, WASAPI_API

ASIO = 4
ASIO_OTHER = 5
WASAPI_MIC = 2
WASAPI_SPEAKERS = 3


class FakeStream:
    def __init__(self, kind, **options):
        self.kind = kind
        self.options = options
        self.active = False
        self.closes = 0
        self.latency = (0.005, 0.006) if kind == "duplex" else 0.006

    def start(self):
        self.active = True

    def abort(self):
        self.active = False

    def close(self):
        self.closes += 1


class FakeSoundDevice:
    def __init__(self):
        self.default = SimpleNamespace(device=(None, None))
        self.opened = []

    @staticmethod
    def WasapiSettings(**options):
        return options

    def query_devices(self, device=None):
        return {"max_input_channels": 2, "max_output_channels": 2}

    def open(self, kind, options):
        stream = FakeStream(kind, **options)
        self.opened.append(stream)
        return stream

    def Stream(self, **options):
        return self.open("duplex", options)

    def InputStream(self, **options):
        return self.open("input", options)

    def OutputStream(self, **options):
        return self.open("output", options)


def endpoint(label, index, api, selectors=()):
    return Endpoint(label, index, api, label, list(selectors))


INPUTS = {
    "asio in": endpoint("asio in", ASIO, ASIO_API, [0, 1]),
    "asio other in": endpoint("asio other in", ASIO_OTHER, ASIO_API, [0, 1]),
    "wasapi mic": endpoint("wasapi mic", WASAPI_MIC, WASAPI_API),
}
OUTPUTS = {
    "asio 1/2": endpoint("asio 1/2", ASIO, ASIO_API, [0, 1]),
    "asio 5/6": endpoint("asio 5/6", ASIO, ASIO_API, [4, 5]),
    "asio other": endpoint("asio other", ASIO_OTHER, ASIO_API, [0, 1]),
    "wasapi speakers": endpoint("wasapi speakers", WASAPI_SPEAKERS, WASAPI_API),
}


class AsioCase(unittest.TestCase):
    """An engine on fake devices; ``self.sd.opened`` lists the streams it opened."""

    def setUp(self):
        self.sd = FakeSoundDevice()
        patcher = mock.patch("engine.core.sd", self.sd)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.events = []

    def engine(self, input_device, output_device, monitor_device=None, sizes=None):
        self.sizes = sizes
        patcher = mock.patch("engine.core.asio_driver.buffer_sizes", lambda sd, device: self.sizes)
        patcher.start()
        self.addCleanup(patcher.stop)
        return self.make_engine(input_device, output_device, monitor_device)

    def make_engine(self, input_device, output_device, monitor_device=None):
        engine = passthrough_engine(block_time=0.04)
        engine.emit = lambda event, data: self.events.append((event, data))
        engine.devices = SimpleNamespace(inputs=INPUTS, outputs=OUTPUTS)
        engine.settings.input_device = input_device
        engine.settings.output_device = output_device
        engine.settings.monitor_device = monitor_device
        engine.samplerate = 48000
        engine.native_worker_stop = None
        engine.native_worker_wakeup = None
        engine.native_worker = None
        engine.asio_sizes = {}
        engine.asio_frames = None
        engine.reported_asio_frames = None
        engine.asio_latency = None
        engine.file_audio_source = None
        engine.stream = engine.input_stream = engine.output_stream = engine.monitor_stream = None
        engine.emit_state = lambda: None
        engine.set_devices()
        engine.input_channels = engine.get_device_channels()
        engine.output_channels = engine.get_output_channels()
        self.addCleanup(engine.stop)
        return engine


class AsioRouteTest(AsioCase):
    def test_input_and_output_on_one_asio_device_share_a_duplex_stream(self):
        engine = self.engine("asio in", "asio 5/6")
        engine.start_stream()

        self.assertEqual([stream.kind for stream in self.sd.opened], ["duplex"])
        stream = self.sd.opened[0]
        self.assertIs(engine.input_stream, stream)
        self.assertIs(engine.output_stream, stream)
        self.assertEqual(stream.options["channels"], (2, 6))
        self.assertEqual(stream.options["device"], (ASIO, ASIO))
        self.assertTrue(engine.streams_active())

        # Input goes to the inference FIFO; queued output goes to channels 5 and 6.
        frames = 64
        engine.output_queue.write(np.full((frames, 1), 0.5, dtype=np.float32))
        indata = np.full((frames, 2), 0.25, dtype=np.float32)
        outdata = np.ones((frames, 6), dtype=np.float32)
        stream.options["callback"](indata, outdata, frames, None, None)
        self.assertEqual(engine.native_input_fifo.read(frames, exact=True).shape, (frames, 2))
        np.testing.assert_array_equal(outdata[:, :4], 0.0)
        np.testing.assert_array_equal(outdata[:, 4:], 0.5)

        engine.stop()
        self.assertEqual(stream.closes, 1)

    def test_monitor_on_the_same_asio_device_shares_the_stream(self):
        engine = self.engine("asio in", "asio 1/2", monitor_device="asio 5/6")
        engine.start_stream()

        self.assertEqual([stream.kind for stream in self.sd.opened], ["duplex"])
        stream = self.sd.opened[0]
        self.assertIs(engine.monitor_stream, stream)
        frames = 32
        engine.output_queue.write(np.full((frames, 1), 0.5, dtype=np.float32))
        engine.monitor_queue.write(np.full((frames, 1), 0.25, dtype=np.float32))
        outdata = np.ones((frames, 6), dtype=np.float32)
        stream.options["callback"](np.zeros((frames, 2), np.float32), outdata, frames, None, None)
        np.testing.assert_array_equal(outdata[:, :2], 0.5)
        np.testing.assert_array_equal(outdata[:, 2:4], 0.0)
        np.testing.assert_array_equal(outdata[:, 4:], 0.25)

    def test_wasapi_input_with_asio_output_and_monitor_opens_one_asio_stream(self):
        engine = self.engine("wasapi mic", "asio 1/2", monitor_device="asio 5/6")
        engine.start_stream()

        self.assertEqual(sorted(stream.kind for stream in self.sd.opened), ["input", "output"])
        asio = next(stream for stream in self.sd.opened if stream.kind == "output")
        self.assertIs(engine.output_stream, asio)
        self.assertIs(engine.monitor_stream, asio)
        self.assertEqual(asio.options["channels"], 6)

    def test_one_asio_side_is_unchanged(self):
        engine = self.engine("asio in", "wasapi speakers")
        engine.start_stream()

        self.assertEqual(sorted(stream.kind for stream in self.sd.opened), ["input", "output"])

    def test_two_asio_devices_are_refused_before_opening(self):
        with self.assertRaises(EngineError) as raised:
            self.engine("asio in", "asio other")
        self.assertEqual(raised.exception.code, "asio_one_device")

    def test_monitor_on_a_second_asio_device_is_turned_off(self):
        engine = self.engine("asio in", "asio 1/2", monitor_device="asio other")

        self.assertIsNone(engine.monitor_device_index)
        self.assertEqual(self.events[-1][1]["code"], "monitor_failed")


class AsioBufferTest(AsioCase):
    SIZES = BufferSizes(minimum=32, maximum=2048, preferred=256, granularity=-1)

    def test_chosen_buffer_is_requested_as_latency(self):
        engine = self.engine("asio in", "asio 1/2", sizes=self.SIZES)
        engine.settings.asio_buffer = 128
        engine.start_stream()

        self.assertAlmostEqual(self.sd.opened[0].options["latency"], 128 / 48000)

    def test_driver_setting_reads_the_drivers_current_size(self):
        engine = self.engine("asio in", "asio 1/2", sizes=self.SIZES)
        engine.start_stream()

        self.assertAlmostEqual(self.sd.opened[0].options["latency"], 256 / 48000)

    def test_without_the_drivers_sizes_the_driver_setting_is_low(self):
        engine = self.engine("asio in", "asio 1/2")
        engine.start_stream()

        self.assertEqual(self.sd.opened[0].options["latency"], "low")

    def test_wasapi_streams_are_not_affected(self):
        engine = self.engine("asio in", "wasapi speakers", sizes=self.SIZES)
        engine.settings.asio_buffer = 64
        engine.start_stream()

        latencies = {stream.kind: stream.options["latency"] for stream in self.sd.opened}
        self.assertEqual(latencies["output"], "low")
        self.assertAlmostEqual(latencies["input"], 64 / 48000)

    def test_buffer_in_use_comes_from_the_callbacks(self):
        engine = self.engine("asio in", "asio 1/2", sizes=self.SIZES)
        engine.start_stream()
        self.assertIsNone(engine.asio_state()["in_use"])

        frames = 192
        stream = self.sd.opened[0]
        stream.options["callback"](
            np.zeros((frames, 2), np.float32), np.zeros((frames, 2), np.float32), frames, None, None
        )
        state = engine.asio_state()
        self.assertEqual(state["in_use"], 192)
        self.assertEqual(state["sizes"]["preferred"], 256)

    def test_no_asio_state_without_an_asio_device(self):
        engine = self.engine("wasapi mic", "wasapi speakers", sizes=self.SIZES)

        self.assertIsNone(engine.asio_state())

    def test_control_panel_is_refused_while_running(self):
        engine = self.engine("asio in", "asio 1/2", sizes=self.SIZES)
        engine.start_stream()

        with self.assertRaises(EngineError) as raised:
            engine.open_asio_control_panel()
        self.assertEqual(raised.exception.code, "asio_panel_busy")


class BufferSizesTest(unittest.TestCase):
    def test_portaudio_rounding(self):
        powers = BufferSizes(minimum=32, maximum=2048, preferred=256, granularity=-1)
        self.assertEqual(powers.accepts(100), 128)
        self.assertEqual(powers.accepts(16), 32)
        self.assertEqual(powers.accepts(4096), 2048)
        steps = BufferSizes(minimum=48, maximum=1024, preferred=256, granularity=48)
        self.assertEqual(steps.accepts(100), 144)
        # Granularity 0: anything between min and max opens the preferred size.
        preferred_only = BufferSizes(minimum=64, maximum=1024, preferred=256, granularity=0)
        self.assertEqual(preferred_only.accepts(128), 256)
        self.assertEqual(preferred_only.accepts(64), 64)
        fixed = BufferSizes(minimum=512, maximum=512, preferred=512, granularity=0)
        self.assertTrue(fixed.fixed)
        self.assertEqual(fixed.accepts(64), 512)

    def test_setting_accepts_only_listed_sizes(self):
        self.assertEqual(coerce_setting("asio_buffer", 128), 128)
        self.assertEqual(coerce_setting("asio_buffer", 128.0), 128)
        for bad in (100, True, "128", 12.5):
            with self.assertRaises(SettingError):
                coerce_setting("asio_buffer", bad)


if __name__ == "__main__":
    unittest.main()
