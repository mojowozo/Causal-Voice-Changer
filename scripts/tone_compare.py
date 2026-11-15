"""Compare voices on laughs and speech: does the output follow the input's tone movement,
and does it still sound like the target (and not like the source)?

    PYTHONPATH=$PWD .venv/bin/python scripts/tone_compare.py OUTDIR TARGET_AUDIO_DIR label=DECODER.pt:SPEAKER ... [-- extra.wav:semitones ...]
"""
import glob, os, sys
import numpy as np, soundfile as sf, torch, torchaudio
sys.path.insert(0, os.path.dirname(__file__))
from probe import DEV, load48
from dynamics import feats  # noqa: E402
from causalvc.infer import CausalVC
from causalvc.voice import shared
from vcbench.metrics.models import SpeakerEmbedder

args = sys.argv[1:]; extra = []
if "--" in args:
    extra = args[args.index("--") + 1:]; args = args[: args.index("--")]
out, tdir, specs = args[0], args[1], args[2:]
emb = SpeakerEmbedder(DEV); r16 = lambda x: torchaudio.functional.resample(x.cpu(), 48000, 16000).numpy()
tf = [f for f in sorted(glob.glob(tdir + "/**/*.*", recursive=True)) if f.lower().endswith((".wav", ".flac"))]
cen = emb.centroid([r16(load48(f)[: 10 * 48000]) for f in tf[:: max(1, len(tf) // 16)][:16]])
SRC = ("p021", "p042", "p063", "p100")
clips = {}
for kind, stem in (("speech", "rainbow_01_regular"), ("laugh open", "nonverbal_laughter_open"), ("laugh closed", "nonverbal_laughter_closed"), ("crying", "nonverbal_crying")):
    for s in SRC:
        clips[(kind, s)] = (load48(f"data/raw/ears/{s}/{stem}.wav")[: 12 * 48000], 0.0)
for e in extra:
    p, st = e.rsplit(":", 1); clips[("yours: " + os.path.basename(p)[:-4], "you")] = (load48(p), float(st))
kinds = list(dict.fromkeys(k for k, _ in clips))
def follow(x, y):
    A, B = feats(x), feats(y); T = min(A.shape[1], B.shape[1]); A, B = A[:, :T], B[:, :T]; act = A[2] > A[2].max() - 35
    return [float(torch.corrcoef(torch.stack([A[i, act], B[i, act]]))[0, 1]) for i in (0, 1)]
print("per kind: brightness follows input / openness follows input / similarity to target / similarity to source")
print(f"{'':16}" + "".join(f"{k:>26}" for k in kinds))
for spec in specs:
    label, rest = spec.split("="); dec, spk = rest.rsplit(":", 1)
    vc = CausalVC(dec, DEV, pitch=shared("pitch.pt")); os.makedirs(f"{out}/{label}", exist_ok=True); res = {k: [] for k in kinds}
    for (kind, s), (x, st) in clips.items():
        torch.manual_seed(0); y = vc.convert(x, spk, semitones=st).cpu()
        sf.write(f"{out}/{label}/{kind.replace(' ', '_').replace(':', '')}__{s}.wav", y.numpy(), 48000, subtype="PCM_16")
        e = emb.embed(r16(y)); res[kind].append(follow(x, y) + [float(e @ cen), float(e @ emb.embed(r16(x)))])
    print(f"{label:16}" + "".join(f"{'  '.join(f'{v:.2f}' for v in np.mean(res[k], 0)):>26}" for k in kinds), flush=True)
