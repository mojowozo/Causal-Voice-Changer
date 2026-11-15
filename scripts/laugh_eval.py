"""Where does laughter go wrong: content, pitch, or the decoder?

    PYTHONPATH=$PWD .venv/bin/python scripts/laugh_eval.py [--speakers 12] [--voice realtime/models/P89 --own p089]

Part 1, per kind of recording (read speech, laughter, ...), over several EARS speakers:
  how closely the streaming content encoder follows its teacher (ContentVec), and
  how closely the streaming pitch tracker follows RMVPE.
Part 2, with a built voice and that speaker's own recordings: convert each recording
  back into the same voice with teacher or streaming parts swapped in, and measure
  the distance to the original. Runs on the CPU unless DEV is set.
"""
import argparse, glob, json, os, sys
import numpy as np, torch, torchaudio

DEV = os.environ.setdefault("DEV", "cpu")
sys.path.insert(0, os.path.dirname(__file__))
from probe import load48
from causalvc.encoder.train import load_teacher
from causalvc.infer import CausalVC, load_encoder
from causalvc.pitch import Rmvpe
from causalvc.pitch_causal import load_pitch
from causalvc.voice import shared

KINDS = {"read speech": "rainbow_01_regular", "free speech": "freeform_speech_01", "whisper": "rainbow_01_whisper",
         "laugh (open)": "nonverbal_laughter_open", "laugh (closed)": "nonverbal_laughter_closed", "crying": "nonverbal_crying"}
ap = argparse.ArgumentParser()
ap.add_argument("--ears", default="data/raw/ears"); ap.add_argument("--speakers", type=int, default=12)
ap.add_argument("--seconds", type=float, default=20); ap.add_argument("--voice"); ap.add_argument("--own")
ap.add_argument("--write")
a = ap.parse_args()
torch.set_num_threads(4)
mel = torchaudio.transforms.MelSpectrogram(48000, 2048, hop_length=480, n_mels=80, f_min=50, f_max=16000)
clip = lambda p: (lambda x: x[: int(a.seconds * 48000) // 960 * 960])(load48(p))
to16 = lambda x: torchaudio.functional.resample(x, 48000, 16000)[None].to(DEV)

teacher, student = load_teacher(DEV), load_encoder(shared("encoder.pt"), DEV)
rmvpe, causal = Rmvpe(DEV), load_pitch(shared("pitch.pt"), DEV)
dirs = sorted(glob.glob(f"{a.ears}/p*")); dirs = dirs[:: max(1, len(dirs) // a.speakers)][: a.speakers]
print(f"part 1: {len(dirs)} speakers, {DEV}")
print(f"{'kind':16} {'content cos':>11} {'worst 10%':>9} | {'voiced':>6} {'pitch<50c':>9} {'octave':>6} {'missed':>6} {'false':>6}")
for kind, stem in KINDS.items():
    cos, cents, n = [], [], np.zeros(4)
    for d in dirs:
        p = f"{d}/{stem}.wav"
        if not os.path.exists(p):
            continue
        w = to16(clip(p))
        with torch.no_grad():
            t = teacher(w).last_hidden_state[0]; s = student.encode(w)[0].float(); T = min(len(t), len(s))
            ft, fs = rmvpe(w)[0], causal(w)[0]; F = min(len(ft), len(fs)); ft, fs = ft[:F], fs[:F]
        act = torch.nn.functional.avg_pool1d(w.abs()[None], 320, 320)[0, 0][:T] > 0.003  # skip silence
        cos.append(torch.nn.functional.cosine_similarity(t[:T], s[:T], dim=-1)[act[:T]].cpu())
        vt, vs = ft > 0, fs > 0; both = vt & vs
        cents.append((1200 * torch.log2(fs[both] / ft[both]).abs()).cpu())
        n += np.array([int(vt.sum()), F, int((vt & ~vs).sum()), int((vs & ~vt).sum())])
    if not cos:
        continue
    c, e = torch.cat(cos), torch.cat(cents)
    print(f"{kind:16} {c.mean():11.3f} {c.quantile(0.1):9.3f} | {n[0] / n[1]:6.0%} {(e < 50).float().mean():9.1%} "
          f"{(e > 600).float().mean():6.1%} {n[2] / max(n[0], 1):6.1%} {n[3] / max(n[1] - n[0], 1):6.1%}")

if a.voice:
    from causalvc.voice import load
    v = load(a.voice); spk = v["speaker"]; print(f"\npart 2: {a.own} -> {spk} (own recordings), log-mel distance to the original, lower is closer")

    class T:  # teacher content with the interface the decoder wrapper expects
        encode = staticmethod(lambda w: teacher(w.float()).last_hidden_state)
    vc = CausalVC(v["decoder"], DEV); enc = vc.enc
    setups = {"teacher content + RMVPE": (T, rmvpe), "streaming content + RMVPE": (enc, rmvpe),
              "teacher content + streaming pitch": (T, causal), "all streaming": (enc, causal)}
    lm = lambda x: torch.log(mel(x) + 1e-5)
    print(f"{'kind':16} " + " ".join(f"{k:>34}" for k in setups))
    for kind, stem in KINDS.items():
        p = f"{a.ears}/{a.own}/{stem}.wav"
        if not os.path.exists(p):
            continue
        x = clip(p); row = []
        for k, (e, f) in setups.items():
            vc.enc, vc.rmvpe = e, f; torch.manual_seed(0)
            y = vc.convert(x, spk).cpu(); row.append(float((lm(y) - lm(x)).abs().mean()))
            if a.write:
                import soundfile as sf
                os.makedirs(a.write, exist_ok=True)
                sf.write(f"{a.write}/{stem}__{k.replace(' ', '_').replace('+', 'and')}.wav", y.numpy(), 48000, subtype="PCM_16")
        print(f"{kind:16} " + " ".join(f"{r:34.3f}" for r in row))
