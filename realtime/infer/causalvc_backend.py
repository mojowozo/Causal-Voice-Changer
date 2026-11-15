"""A CausalVC voice for the realtime engine.

It converts each block as it arrives and its output is already continuous
across blocks.  A voice is a folder in ``models/`` holding its description and
its decoder, as written when a voice is built:

    MyVoice-10000s/
      MyVoice-10000s.causalvc.json     {"decoder": "decoder.pt", "speaker": "MyVoice"}
      decoder.pt

The folder is self-contained (see ``causalvc.voice``); the content encoder and
pitch tracker every voice shares come with the project's base.  ``"runtime":
"classic"`` in the description selects the original runtime, which recomputes
its whole window every block.
"""

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
        if REPOSITORY not in sys.path:  # the causalvc package, when it is not installed in this environment
            sys.path.insert(0, REPOSITORY)
        from causalvc.stream import DELAY
        from causalvc.voice import load

        # Paths in the description are looked for beside it first, so a voice folder works wherever it is copied.
        spec = load(model_path)
        if spec["runtime"] == "classic":  # the original runtime, which recomputes its whole window every block
            from causalvc.stream import StreamingVC as make
        else:  # the cached runtime: same model, computes only what is new each block
            from causalvc.fast import streaming as make

        self.speaker = spec["speaker"]
        self.vc = make(spec["decoder"], spec["pitch"], str(device), spec["encoder"])
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
