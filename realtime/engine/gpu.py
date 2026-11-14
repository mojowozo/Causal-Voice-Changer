"""Which device the model runs on."""

import torch


def cuda_indices():
    """Indices of the CUDA devices PyTorch can use."""
    return list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []


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
