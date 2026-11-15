"""Which device the model runs on.

PyTorch's ROCm builds (AMD) use the same ``cuda`` device type and
``torch.cuda`` API as its CUDA builds (NVIDIA), so everything here covers both.
"""

import torch


def backend_label():
    """``"ROCm 7.2"``, ``"CUDA 13.2"``, or ``"CPU only"``: the GPU runtime PyTorch was built with."""
    hip = getattr(torch.version, "hip", None)
    if hip:
        return f"ROCm {'.'.join(hip.split('.')[:2])}"
    if torch.version.cuda:
        return f"CUDA {torch.version.cuda}"
    return "CPU only"


def cuda_indices():
    """Indices of the GPUs PyTorch can use (NVIDIA through CUDA, AMD through ROCm)."""
    return list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []


def gpu_name(index):
    return torch.cuda.get_device_name(index)


def default_index():
    """The GPU "auto" means: the one with the most memory, or None without a GPU."""
    indices = cuda_indices()
    if not indices:
        return None
    return max(indices, key=lambda index: torch.cuda.get_device_properties(index).total_memory)


class Device:
    """The selected device, as a string PyTorch accepts ("cuda:0" or "cpu")."""

    def __init__(self):
        index = default_index()
        self.device = "cpu" if index is None else f"cuda:{index}"

    def select_cuda_device(self, index):
        index = int(index)
        if index not in cuda_indices():
            raise ValueError(f"GPU {index} is not available")
        self.device = f"cuda:{index}"
