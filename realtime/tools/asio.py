"""ASIO driver helpers that sounddevice does not wrap: the driver's buffer sizes
and its control panel (PortAudio's PaAsio_* functions).

They are called on the PortAudio library sounddevice already loaded, so they
share its state.  Anything missing (no ASIO build, an older library) reads as
"not available" rather than an error.
"""

import ctypes
from dataclasses import dataclass


@dataclass(frozen=True)
class BufferSizes:
    """What the driver accepts, in samples.  ``granularity`` is 0 when the
    driver only takes min, max or its preferred size, -1 for powers of two."""

    minimum: int
    maximum: int
    preferred: int
    granularity: int

    @property
    def fixed(self):
        return self.minimum == self.maximum

    def accepts(self, frames):
        """The size PortAudio will open for a request of ``frames``
        (pa_asio.cpp: SelectHostBufferSizeForUnspecifiedUserFramesPerBuffer)."""
        if frames <= self.minimum:
            return self.minimum
        if frames >= self.maximum:
            return self.maximum
        if self.granularity == 0:
            return self.preferred
        if self.granularity == -1:
            size = 1
            while size < frames:
                size *= 2
        else:
            size = -(-frames // self.granularity) * self.granularity
        return min(max(size, self.minimum), self.maximum)

    def to_dict(self):
        return {
            "min": self.minimum,
            "max": self.maximum,
            "preferred": self.preferred,
            "granularity": self.granularity,
        }


def _library(sd):
    path = getattr(sd, "_libname", None)
    if not path:
        return None
    try:
        return ctypes.CDLL(path)
    except OSError:
        return None


def buffer_sizes(sd, device):
    """The driver's buffer sizes for PortAudio device ``device``, or None."""
    library = _library(sd)
    function = getattr(library, "PaAsio_GetAvailableBufferSizes", None)
    if function is None:
        return None
    function.restype = ctypes.c_int
    values = [ctypes.c_long() for _ in range(4)]
    try:
        error = function(ctypes.c_int(device), *(ctypes.byref(value) for value in values))
    except OSError:
        return None
    if error != 0:
        return None
    minimum, maximum, preferred, granularity = (value.value for value in values)
    if minimum <= 0 or maximum < minimum:
        return None
    return BufferSizes(minimum, maximum, preferred, granularity)


def show_control_panel(sd, device, window=None):
    """Open the driver's control panel; False when it could not be opened
    (no such function, or the driver is in use by an open stream)."""
    library = _library(sd)
    function = getattr(library, "PaAsio_ShowControlPanel", None)
    if function is None:
        return False
    function.restype = ctypes.c_int
    function.argtypes = (ctypes.c_int, ctypes.c_void_p)
    try:
        return function(device, window) == 0
    except OSError:
        return False
