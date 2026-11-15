"""Convert a voice's own recordings back into that voice and measure the distance to the originals.

    PYTHONPATH=$PWD .venv/bin/python scripts/self_recon.py EARS_SPEAKER label=DECODER.pt:SPEAKER[:ENCODER.pt] ...

Log-mel distance (lower is closer) per kind of recording, streaming encoder and pitch tracker.
"""
import os, sys
import torch, torchaudio
sys.path.insert(0, os.path.dirname(__file__))
from probe import DEV, load48
from causalvc.infer import CausalVC
from causalvc.voice import shared

KINDS = {"speech": "rainbow_01_regular", "free": "freeform_speech_01", "whisper": "rainbow_01_whisper", "laugh open": "nonverbal_laughter_open",
         "laugh closed": "nonverbal_laughter_closed", "crying": "nonverbal_crying", "yelling": "nonverbal_yelling"}
mel = torchaudio.transforms.MelSpectrogram(48000, 2048, hop_length=480, n_mels=80, f_min=50, f_max=16000)
lm = lambda x: torch.log(mel(x) + 1e-5)
clips = {k: load48(f"data/raw/ears/{sys.argv[1]}/{s}.wav")[: 20 * 48000] for k, s in KINDS.items()}
print(f"{'':28}" + "".join(f"{k:>14}" for k in clips))
for spec in sys.argv[2:]:
    label, rest = spec.split("="); p = rest.split(":")
    vc = CausalVC(p[0], DEV, encoder=p[2] if len(p) > 2 else None, pitch=shared("pitch.pt")); row = []
    for x in clips.values():
        d = []
        for seed in range(2):
            torch.manual_seed(seed); d.append(float((lm(vc.convert(x, p[1]).cpu()) - lm(x)).abs().mean()))
        row.append(sum(d) / 2)
    print(f"{label:28}" + "".join(f"{r:14.3f}" for r in row), flush=True)
