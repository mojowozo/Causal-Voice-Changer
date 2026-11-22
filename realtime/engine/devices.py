"""PortAudio device catalog: JACK (PipeWire) and PipeWire's ALSA PCMs on Linux,
WASAPI and ASIO on Windows (see ``tools.audio_routing``)."""

from dataclasses import dataclass, field

from tools.audio_routing import (
    CHANNEL_PAIR_APIS,
    HOSTAPI_LABELS,
    WASAPI_API,
    api_priority,
    is_offered,
)


@dataclass
class Endpoint:
    label: str
    index: int
    api: str
    name: str
    selectors: list = field(default_factory=list)


def channel_pairs(channel_count):
    pairs = []
    for first in range(0, int(channel_count), 2):
        second = min(first + 1, int(channel_count) - 1)
        pairs.append(f"{first + 1} / {second + 1}")
    return pairs or ["1 / 1"]


def channel_choices(channel_count):
    """Each channel pair, followed by its two channels on their own ("1 / 2", "1", "2", ...).

    A single channel is for a mic on one input of an interface: with the pair, the empty
    channel beside it would be mixed in and halve its level. A device with one channel (and
    an odd device's last "n / n") is already a single channel.
    """
    choices = []
    for pair in channel_pairs(channel_count):
        choices.append(pair)
        first, second = parse_channel_pair(pair)
        if first != second:
            choices += [str(first + 1), str(second + 1)]
    return choices


def parse_channel_pair(value):
    """"1 / 2" -> [0, 1]; a single channel, "3" -> [2]."""
    return [int(part.strip()) - 1 for part in value.split("/")]


def endpoint_label(api_name, device_name):
    return f"[{HOSTAPI_LABELS.get(api_name, api_name)}] {device_name}"


def label_sort_key(endpoint):
    """Best API first, then by device; a device's channel choices keep their own order."""
    return api_priority(endpoint.api), endpoint_label(endpoint.api, endpoint.name).lower()


def build_endpoints(devices, hostapis, show_all=False):
    """Return (inputs, outputs) as label -> Endpoint, best API first.

    Raw ALSA hw:* devices are normally held open by PipeWire, so only its
    shared PCMs are offered unless ``show_all`` is set; on Windows only WASAPI
    and ASIO are.  JACK and ASIO devices are split into one endpoint per channel
    pair and per channel; a device's first pair keeps its plain label when it has
    no others, so labels saved before single channels were offered still match.
    """
    api_names = {}
    for hostapi in hostapis:
        for device_index in hostapi["devices"]:
            api_names[device_index] = hostapi["name"]
    inputs = {}
    outputs = {}
    for index, device in enumerate(devices):
        api_name = api_names.get(index, "")
        if not show_all and not is_offered(api_name, device["name"]):
            continue
        for max_key, endpoints in (
            ("max_input_channels", inputs),
            ("max_output_channels", outputs),
        ):
            channel_count = int(device[max_key])
            if channel_count <= 0:
                continue
            label = endpoint_label(api_name, device["name"])
            if api_name not in CHANNEL_PAIR_APIS:
                endpoints[label] = Endpoint(label, index, api_name, device["name"])
                continue
            only_pair = len(channel_pairs(channel_count)) == 1
            for choice in channel_choices(channel_count):
                plain = only_pair and "/" in choice
                choice_label = label if plain else f"{label} — {choice}"
                endpoints[choice_label] = Endpoint(
                    choice_label,
                    index,
                    api_name,
                    device["name"],
                    parse_channel_pair(choice),
                )
    return (
        dict(sorted(inputs.items(), key=lambda item: label_sort_key(item[1]))),
        dict(sorted(outputs.items(), key=lambda item: label_sort_key(item[1]))),
    )


def default_devices(sd, hostapis):
    """(input, output) PortAudio indices of the system default devices.

    On Windows PortAudio's own default is the MME device, which is not
    offered, so the WASAPI host API's defaults are used when it has them.
    """
    for hostapi in hostapis:
        if hostapi["name"] == WASAPI_API:
            pair = (hostapi.get("default_input_device", -1), hostapi.get("default_output_device", -1))
            if pair[0] >= 0 or pair[1] >= 0:
                fallback = tuple(sd.default.device)
                return tuple(index if index >= 0 else fallback[i] for i, index in enumerate(pair))
    return tuple(sd.default.device)


class DeviceCatalog:
    def __init__(self, sd):
        self.sd = sd
        self.inputs = {}
        self.outputs = {}
        self.defaults = (None, None)

    def refresh(self, show_all=False):
        # Re-initialize PortAudio so devices added since startup appear.
        self.sd._terminate()
        self.sd._initialize()
        hostapis = self.sd.query_hostapis()
        self.inputs, self.outputs = build_endpoints(self.sd.query_devices(), hostapis, show_all)
        self.defaults = default_devices(self.sd, hostapis)

    def default_label(self, direction):
        endpoints = self.inputs if direction == "input" else self.outputs
        default_index = self.defaults[0 if direction == "input" else 1]
        return next(
            (label for label, ep in endpoints.items() if ep.index == default_index),
            next(iter(endpoints), ""),
        )

    def resolve(self, saved_label, direction):
        """Map a saved label to a current one, falling back to the default.

        A label can disappear when channel counts change, so a device with
        the same PortAudio name is preferred over the system default.
        """
        endpoints = self.inputs if direction == "input" else self.outputs
        if saved_label in endpoints:
            return saved_label
        if saved_label:
            saved_name = saved_label.split("] ", 1)[-1].split(" — ", 1)[0]
            match = next(
                (label for label, ep in endpoints.items() if ep.name == saved_name),
                None,
            )
            if match:
                return match
        return self.default_label(direction)

    def resolve_monitor(self, saved_label):
        if saved_label and saved_label in self.outputs:
            return saved_label
        return None

    def to_dict(self):
        def describe(endpoints):
            return [
                {
                    "label": ep.label,
                    "api": HOSTAPI_LABELS.get(ep.api, ep.api),
                    "name": ep.name,
                    "channels": [selector + 1 for selector in ep.selectors],
                }
                for ep in endpoints.values()
            ]

        return {"inputs": describe(self.inputs), "outputs": describe(self.outputs)}
