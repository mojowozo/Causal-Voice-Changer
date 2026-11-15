"""PortAudio host-API names and channel-selector helpers for routing.

Linux: JACK (PipeWire) and PipeWire's shared ALSA PCMs.  Windows: WASAPI and,
when sounddevice's ASIO build is loaded (``SD_ENABLE_ASIO``), ASIO; MME,
DirectSound and WDM-KS are only listed with ``show_all``.
"""

JACK_API = "JACK Audio Connection Kit"
ALSA_API = "ALSA"
WASAPI_API = "Windows WASAPI"
ASIO_API = "ASIO"
WDMKS_API = "Windows WDM-KS"
MME_API = "MME"
DIRECTSOUND_API = "Windows DirectSound"
HOSTAPI_LABELS = {
    JACK_API: "JACK",
    ALSA_API: "ALSA",
    WASAPI_API: "WASAPI",
    ASIO_API: "ASIO",
    WDMKS_API: "WDM-KS",
    MME_API: "MME",
    DIRECTSOUND_API: "DirectSound",
}
# PipeWire's shared ALSA PCMs.  Raw hw:* devices are held open by PipeWire.
ALSA_SHARED_PCMS = ("pipewire", "default", "pulse")
#: APIs whose devices are offered without ``show_all``, best first.
PREFERRED_APIS = (JACK_API, ALSA_API, ASIO_API, WASAPI_API)
#: APIs whose multi-channel devices are split into one endpoint per channel pair.
CHANNEL_PAIR_APIS = (JACK_API, ASIO_API)


def is_native_api(api_name):
    """JACK and ASIO own their period size, so their streams must use blocksize=0."""
    return api_name in (JACK_API, ASIO_API)


def is_offered(api_name, device_name):
    """Whether a device is listed without ``show_all``."""
    if api_name == ALSA_API:
        return device_name in ALSA_SHARED_PCMS
    return api_name in PREFERRED_APIS


def api_priority(api_name):
    return PREFERRED_APIS.index(api_name) if api_name in PREFERRED_APIS else len(PREFERRED_APIS)


def stream_options(sd, api_name):
    """Extra keyword arguments for a stream (and its settings check) on ``api_name``.

    WASAPI shared mode only opens at the Windows mixer's rate unless asked to
    convert, and the model needs 48 kHz; its default ("high") latency also adds
    tens of milliseconds, so ask for the low one.  ASIO's latency is the
    driver's buffer size, set in its own control panel.
    """
    if api_name == WASAPI_API:
        return {"extra_settings": sd.WasapiSettings(auto_convert=True), "latency": "low"}
    if api_name == ASIO_API:
        return {"latency": "low"}
    return {}


def check_options(options):
    """The part of ``stream_options`` that ``check_*_settings`` accepts."""
    return {key: value for key, value in options.items() if key == "extra_settings"}


def select_channels(indata, selectors):
    """Pick the selected input columns; no selectors keeps every column."""
    if not selectors:
        return indata
    return indata[:, selectors]


def scatter_mono(outdata, mono, selectors):
    """Write a mono block to the selected output columns (or all of them)."""
    sample_count = mono.shape[0]
    if selectors:
        outdata[:sample_count, selectors] = mono[:, None]
    else:
        outdata[:sample_count, :] = mono[:, None]
