"""Convert held-out speakers' laughter (and speech, for reference) with several voice/encoder
pairs, as realtime would (streaming encoder and pitch tracker), and score each.

    PYTHONPATH=$PWD .venv/bin/python scripts/laugh_convert.py OUTDIR label=DECODER.pt:SPEAKER[:ENCODER.pt] ...

Score: how similar the full-size content model (ContentVec) finds input and output,
frame by frame over non-silent audio. Higher means the output still "says" the same
sounds as the input. It does not judge tone or naturalness: listen to the files.
"""
import os, sys
import soundfile as sf, torch, torchaudio
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from probe import DEV, load48
from causalvc.encoder.train import load_teacher
from causalvc.infer import CausalVC
from causalvc.voice import shared

SOURCES = ("p021", "p042", "p063", "p100")
KINDS = {"speech": "rainbow_01_regular", "whisper": "rainbow_01_whisper", "laugh open": "nonverbal_laughter_open",
         "laugh closed": "nonverbal_laughter_closed", "crying": "nonverbal_crying"}
out, specs = sys.argv[1], sys.argv[2:]
extra = [a for a in specs if "=" not in a]; specs = [a for a in specs if "=" in a]  # extra: more input recordings
teacher = load_teacher(DEV)
to16 = lambda x: torchaudio.functional.resample(x.to(DEV), 48000, 16000)[None]
inputs = {(k, s): f"data/raw/ears/{s}/{stem}.wav" for k, stem in KINDS.items() for s in SOURCES}
inputs.update({("own", os.path.splitext(os.path.basename(p))[0]): p for p in extra})
clips = {key: load48(p)[: 15 * 48000] for key, p in inputs.items() if os.path.exists(p)}
kinds = list(dict.fromkeys(k for k, _ in clips))
print(f"{'':28}" + "".join(f"{k:>14}" for k in kinds))
for spec in specs:
    label, rest = spec.split("="); parts = rest.split(":")
    vc = CausalVC(parts[0], DEV, encoder=parts[2] if len(parts) > 2 else None, pitch=shared("pitch.pt"))
    os.makedirs(f"{out}/{label}", exist_ok=True); score = {k: [] for k in kinds}
    for (kind, src), x in clips.items():
        torch.manual_seed(0); y = vc.convert(x, parts[1]).cpu()
        with torch.no_grad():
            a, b = teacher(to16(x)).last_hidden_state[0], teacher(to16(y)).last_hidden_state[0]
        n = min(len(a), len(b)); act = F.avg_pool1d(to16(x).abs()[None], 320, 320)[0, 0][:n] > 0.003
        score[kind].append(F.cosine_similarity(a[:n], b[:n], dim=-1)[act].mean().item())
        sf.write(f"{out}/{label}/{kind.replace(' ', '_')}__{src}.wav", y.numpy(), 48000, subtype="PCM_16")
    print(f"{label:28}" + "".join(f"{sum(v) / len(v):14.3f}" for v in score.values()), flush=True)
os.makedirs(f"{out}/input", exist_ok=True)
for (kind, src), x in clips.items():
    sf.write(f"{out}/input/{kind.replace(' ', '_')}__{src}.wav", x.numpy(), 48000, subtype="PCM_16")
