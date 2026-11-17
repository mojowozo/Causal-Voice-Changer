"""Checks that PyTorch loads, and says what to do when it doesn't.

run-realtime.bat and run-training.bat run this after installing the packages. With
AMD's ROCm build on an old AMD driver or graphics ROCm doesn't support, `import torch`
fails with a bare Windows DLL error ("A dynamic link library (DLL) initialization
routine failed"); this says what to do instead. Exits 1 when PyTorch cannot load.

    python -m causalvc.torch_check
"""

from __future__ import annotations

import sys

#: The AMD driver ROCm 7.2.1 needs (pyproject.toml's rocm extra).
AMD_DRIVER = "AMD Software: Adrenalin Edition 26.2.2"


def rocm_advice() -> str:
    return (
        "AMD's GPU runtime (ROCm) could not start, so PyTorch cannot load.\n"
        f"Most often the AMD graphics driver is older than ROCm needs: install {AMD_DRIVER} or newer\n"
        "(https://www.amd.com/en/support/download/drivers.html), restart Windows, and run this again.\n"
        "ROCm runs on Radeon RX 6000, 7000 and 9000, Radeon PRO W6000/W7000, and Ryzen AI Max / AI 300\n"
        "graphics (the RX 6000 series through AMD's nightly build, which the launcher picks for it).\n"
        "On other graphics, or to skip the GPU, open a Command Prompt in this folder and run\n"
        "    set CAUSALVC_GPU=cuda\n"
        "then run-realtime.bat (or run-training.bat) from that window: voices then run on the processor (slowly)."
    )


def is_rocm_failure(text: str) -> bool:
    """A start-up error that came from loading AMD's ROCm runtime."""
    text = text.lower()
    return "rocm" in text and ("winerror 1114" in text or "dll" in text)


def installed_torch() -> str:
    """The installed PyTorch's version, read without importing it."""
    from importlib import metadata

    try:
        return metadata.version("torch")
    except metadata.PackageNotFoundError:
        return "none"


def main() -> int:
    try:
        import torch  # noqa: F401
    except OSError as error:
        import traceback

        detail = traceback.format_exc()
        print(detail, file=sys.stderr)
        if is_rocm_failure(detail):
            print(rocm_advice(), file=sys.stderr)
        else:
            print(f"PyTorch could not load: {error}", file=sys.stderr)
        print(f"PyTorch build installed: {installed_torch()} (in {sys.prefix})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
