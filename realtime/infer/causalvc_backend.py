"""A CausalVC voice for the realtime engine.

It converts each block as it arrives and its output is already continuous
across blocks.  A voice is a folder in ``models/`` holding
``<anything>.causalvc.json``, as written when a voice is built:

    {"decoder": "runs/models/voice-NAME/decoder-10000s.pt",
     "pitch":   "weights/causalvc-base/pitch.pt",
     "speaker": "NAME"}

Relative paths are resolved against ``project`` when the file names one, and
against the repository this app lives in otherwise.  ``encoder`` is optional
(the decoder checkpoint names the one it was trained with).  ``"runtime":
"classic"`` selects the original runtime, which recomputes its whole window
every block.
"""

import json
import os
import sys

import torch

#: Voice descriptions; the second spelling is the project's earlier name and still loads.
SUFFIXES = (".causalvc.json", ".nextvc.json")
#: Blocks must be a whole number of 20 ms encoder frames at 48 kHz.
SAMPLE_RATE = 48000
#: The repository this app is part of (the folder above realtime/).
REPOSITORY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BLOCK_MULTIPLE = 960


def is_causalvc(model_path):
    return str(model_path).endswith(SUFFIXES)


class CausalVC:
    is_causalvc = True
    tgt_sr = SAMPLE_RATE

    def __init__(self, key, model_path, device):
        with open(model_path, encoding="utf-8") as handle:
            spec = json.load(handle)
        project = spec.get("project") or REPOSITORY
        if project not in sys.path:  # the causalvc package, when it is not installed in this environment
            sys.path.insert(0, project)
        from causalvc.stream import DELAY

        if spec.get("runtime") == "classic":  # the original runtime, which recomputes its whole window every block
            from causalvc.stream import StreamingVC as make
        else:  # the cached runtime: same model, computes only what is new each block
            from causalvc.fast import streaming as make

        def resolve(path):
            return path if path is None or os.path.isabs(path) else os.path.join(project, path)

        self.speaker = spec["speaker"]
        self.vc = make(
            resolve(spec["decoder"]), resolve(spec["pitch"]), str(device), resolve(spec.get("encoder"))
        )
        if self.speaker not in self.vc.speakers:
            raise ValueError(f"{self.speaker} is not a voice of {spec['decoder']}")
        self.delay_seconds = getattr(self.vc, "delay", DELAY) / 100.0
        self.change_key(key)

    def change_key(self, key):
        self.vc.set_voice(self.speaker, float(key))

    def reset_stream_state(self):
        self.vc.reset()

    def process(self, block):
        """One block of 48 kHz mono samples (a torch tensor) -> the same length."""
        return self.vc.process(block)

    def warm(self, block_frame):
        for _ in range(4):
            self.vc.process(torch.zeros(block_frame, device=self.vc.device))
        self.vc.reset()
