import unittest
from types import SimpleNamespace

from engine.devices import DeviceCatalog, build_endpoints

HOSTAPIS = [
    {"name": "ALSA", "devices": [0, 1, 2]},
    {"name": "JACK Audio Connection Kit", "devices": [3, 4, 5]},
]
DEVICES = [
    {"name": "HD-Audio Generic: ALC897 Analog (hw:3,0)", "max_input_channels": 2, "max_output_channels": 6},
    {"name": "pipewire", "max_input_channels": 128, "max_output_channels": 128},
    {"name": "default", "max_input_channels": 128, "max_output_channels": 128},
    {"name": "Elgato Wave XLR Mono", "max_input_channels": 1, "max_output_channels": 0},
    {"name": "Creative BT-W6 Analog Stereo", "max_input_channels": 2, "max_output_channels": 2},
    {"name": "GB207 Pro", "max_input_channels": 0, "max_output_channels": 8},
]


class BuildEndpointsTest(unittest.TestCase):
    def test_hides_raw_alsa_devices_and_lists_jack_first(self):
        inputs, outputs = build_endpoints(DEVICES, HOSTAPIS)

        self.assertNotIn("[ALSA] HD-Audio Generic: ALC897 Analog (hw:3,0)", inputs)
        self.assertEqual(
            list(inputs),
            [
                "[JACK] Creative BT-W6 Analog Stereo",
                "[JACK] Creative BT-W6 Analog Stereo — 1",
                "[JACK] Creative BT-W6 Analog Stereo — 2",
                "[JACK] Elgato Wave XLR Mono",
                "[ALSA] default",
                "[ALSA] pipewire",
            ],
        )
        self.assertTrue(list(outputs)[0].startswith("[JACK]"))

    def test_show_all_includes_raw_alsa_devices(self):
        inputs, _ = build_endpoints(DEVICES, HOSTAPIS, show_all=True)

        self.assertIn("[ALSA] HD-Audio Generic: ALC897 Analog (hw:3,0)", inputs)

    def test_jack_channel_pairs_and_selectors(self):
        inputs, outputs = build_endpoints(DEVICES, HOSTAPIS)

        self.assertEqual(inputs["[JACK] Elgato Wave XLR Mono"].selectors, [0, 0])
        self.assertEqual(inputs["[JACK] Creative BT-W6 Analog Stereo"].selectors, [0, 1])
        self.assertEqual(outputs["[JACK] GB207 Pro — 3 / 4"].selectors, [2, 3])
        self.assertEqual(outputs["[JACK] GB207 Pro — 7 / 8"].index, 5)
        self.assertNotIn("[JACK] GB207 Pro", outputs)
        self.assertEqual(outputs["[ALSA] pipewire"].selectors, [])


class FakeSoundDevice:
    def __init__(self):
        self.default = SimpleNamespace(device=(2, 2))

    def _terminate(self):
        pass

    def _initialize(self):
        pass

    def query_devices(self):
        return DEVICES

    def query_hostapis(self):
        return HOSTAPIS


class DeviceCatalogTest(unittest.TestCase):
    def setUp(self):
        self.catalog = DeviceCatalog(FakeSoundDevice())
        self.catalog.refresh()

    def test_resolve_keeps_existing_label(self):
        label = "[JACK] Elgato Wave XLR Mono"
        self.assertEqual(self.catalog.resolve(label, "input"), label)

    def test_resolve_matches_same_device_when_pair_label_changes(self):
        self.assertEqual(
            self.catalog.resolve("[JACK] GB207 Pro", "output"),
            "[JACK] GB207 Pro — 1 / 2",
        )

    def test_resolve_falls_back_to_portaudio_default(self):
        self.assertEqual(self.catalog.resolve("[JACK] Unplugged", "input"), "[ALSA] default")
        self.assertEqual(self.catalog.resolve("", "output"), "[ALSA] default")

    def test_missing_monitor_is_disabled(self):
        self.assertIsNone(self.catalog.resolve_monitor("[JACK] Unplugged"))
        self.assertIsNone(self.catalog.resolve_monitor(None))

    def test_to_dict_reports_one_based_channels(self):
        outputs = {item["label"]: item for item in self.catalog.to_dict()["outputs"]}

        self.assertEqual(outputs["[JACK] GB207 Pro — 3 / 4"]["channels"], [3, 4])
        self.assertEqual(outputs["[JACK] GB207 Pro — 3 / 4"]["api"], "JACK")


WINDOWS_HOSTAPIS = [
    {"name": "MME", "devices": [0, 1], "default_input_device": 0, "default_output_device": 1},
    {"name": "Windows WASAPI", "devices": [2, 3], "default_input_device": 2, "default_output_device": 3},
    {"name": "ASIO", "devices": [4], "default_input_device": 4, "default_output_device": 4},
]
WINDOWS_DEVICES = [
    {"name": "Microphone (Wave XLR)", "max_input_channels": 2, "max_output_channels": 0},
    {"name": "Speakers (Realtek)", "max_input_channels": 0, "max_output_channels": 2},
    {"name": "Microphone (Wave XLR)", "max_input_channels": 1, "max_output_channels": 0},
    {"name": "Speakers (Realtek)", "max_input_channels": 0, "max_output_channels": 2},
    {"name": "Focusrite USB ASIO", "max_input_channels": 2, "max_output_channels": 4},
]


class WindowsSoundDevice(FakeSoundDevice):
    def __init__(self):
        self.default = SimpleNamespace(device=(0, 1))  # PortAudio's default is MME

    def query_devices(self):
        return WINDOWS_DEVICES

    def query_hostapis(self):
        return WINDOWS_HOSTAPIS


class WindowsDevicesTest(unittest.TestCase):
    def test_offers_asio_then_wasapi_and_hides_mme(self):
        inputs, outputs = build_endpoints(WINDOWS_DEVICES, WINDOWS_HOSTAPIS)

        self.assertEqual(
            list(inputs),
            [
                "[ASIO] Focusrite USB ASIO",
                "[ASIO] Focusrite USB ASIO — 1",
                "[ASIO] Focusrite USB ASIO — 2",
                "[WASAPI] Microphone (Wave XLR)",
            ],
        )
        self.assertEqual(
            list(outputs),
            [
                "[ASIO] Focusrite USB ASIO — 1 / 2",
                "[ASIO] Focusrite USB ASIO — 1",
                "[ASIO] Focusrite USB ASIO — 2",
                "[ASIO] Focusrite USB ASIO — 3 / 4",
                "[ASIO] Focusrite USB ASIO — 3",
                "[ASIO] Focusrite USB ASIO — 4",
                "[WASAPI] Speakers (Realtek)",
            ],
        )
        self.assertEqual(outputs["[ASIO] Focusrite USB ASIO — 3 / 4"].selectors, [2, 3])
        # One channel on its own: a mic on input 1 is not mixed with an empty input 2.
        self.assertEqual(inputs["[ASIO] Focusrite USB ASIO"].selectors, [0, 1])
        self.assertEqual(inputs["[ASIO] Focusrite USB ASIO — 2"].selectors, [1])
        self.assertEqual(outputs["[ASIO] Focusrite USB ASIO — 3"].selectors, [2])
        self.assertEqual(outputs["[WASAPI] Speakers (Realtek)"].selectors, [])
        shown_all, _ = build_endpoints(WINDOWS_DEVICES, WINDOWS_HOSTAPIS, show_all=True)
        self.assertIn("[MME] Microphone (Wave XLR)", shown_all)

    def test_defaults_come_from_wasapi_not_mme(self):
        catalog = DeviceCatalog(WindowsSoundDevice())
        catalog.refresh()

        self.assertEqual(catalog.resolve("", "input"), "[WASAPI] Microphone (Wave XLR)")
        self.assertEqual(catalog.resolve("[WASAPI] Unplugged", "output"), "[WASAPI] Speakers (Realtek)")


if __name__ == "__main__":
    unittest.main()
