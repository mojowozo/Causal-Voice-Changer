"""Tone, clarity and similarity of a built voice against the target's real recording.

    PYTHONPATH=$PWD .venv/bin/python scripts/tone_eval.py TARGET.wav SPEAKER label=decoder.pt [...]

For each decoder: copy-synthesis of the target's own audio, and conversion of the
benchmark sources (shifted to the target's median pitch), each as dB per band
relative to the real recording (bands 80-250-500-1k-2k-4k-8k-16k Hz).
"""
import json, os, sys
import numpy as np, torch, torchaudio
sys.path.insert(0, os.path.dirname(__file__))
from probe import DEV, clarity, load48, rmvpe_f0, tone
from causalvc.decoder.model import FRAME
from causalvc.infer import CausalVC, extract
from vcbench.metrics.models import SpeakerEmbedder

emb = SpeakerEmbedder(DEV)
r16 = lambda x: torchaudio.functional.resample(x.cpu(), 48000, 16000).numpy()
tgt = load48(sys.argv[1]); name = sys.argv[2]; n = 48000 * 8
real = [tgt[i:i + n] for i in np.linspace(0, len(tgt) - n, 16).astype(int)]
rb = tone(real); cen = emb.centroid([r16(x) for x in real])
medf = lambda x: float((lambda f: f[f > 0].median())(rmvpe_f0(x)))
tf = float(np.median([medf(x) for x in real[:8]]))
m = json.load(open("data/eval/manifest.json")); by = {}
for s in m["sources"]:
    by.setdefault(s["id"].split("_")[0], []).append(s["wav"])
srcs = [(load48(p), k) for k, v in by.items() for p in v[:5]]
ratio = {k: tf / np.median([medf(x) for x, kk in srcs if kk == k]) for k in by}
print(f"target median f0 {tf:.0f} Hz, real clarity {np.round(np.mean([c for c in (clarity(x)[0] for x in real) if c], 0), 1)}")
for a in sys.argv[3:]:
    label, dec = a.split("=")
    vc = CausalVC(dec, DEV); d = vc.dec; spk = torch.tensor([vc.speakers.index(name)], device=DEV)

    @torch.no_grad()
    def syn(x, fmul=1.0):
        x = x[: len(x) // (2 * FRAME) * 2 * FRAME].to(DEV)[None]
        c, f, l = extract(vc.enc, vc.rmvpe, x); f = f * fmul
        h, g = d.features(c, f, l, spk)
        return d.generate(h, g, d.source(f[:, : h.shape[-1]]))[0, 0].cpu()

    copy = [syn(x) for x in real]; conv = [syn(x, ratio[k]) for x, k in srcs]
    print(f"{label}\n   copy-synthesis tone {np.round(tone(copy) - rb, 1)}\n   conversion tone     {np.round(tone(conv) - rb, 1)}"
          f"\n   conversion similarity {np.mean([emb.embed(r16(y)) @ cen for y in conv]):.3f}, clarity {np.round(np.mean([c for c in (clarity(y)[0] for y in conv) if c], 0), 1)}")
