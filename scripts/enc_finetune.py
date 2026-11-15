"""Experiment: can the streaming content encoder learn laughter if it is shown some?

    PYTHONPATH=$PWD .venv/bin/python scripts/enc_finetune.py --name enc-la2-ft-expressive --device cuda:1

Continues the released encoder on a mix of audiobook reading (LibriTTS-R), VCTK and
EARS (which has laughing, crying, whispering, shouting), against the same teacher.
A few EARS speakers are held out and scored separately on speech and on laughter.
"""
import argparse, json, random, time
from pathlib import Path
import numpy as np, soundfile as sf, torch, torchaudio
import torch.nn.functional as F

from causalvc.encoder.model import HOP, SR, StreamingEncoder
from causalvc.encoder.train import distill_loss, load_teacher, teacher_features
from causalvc.voice import shared

HELD = ("p021", "p042", "p063", "p089", "p100")
EXPRESSIVE = ("nonverbal_", "_whisper", "interjection_")


class Mixed(torch.utils.data.IterableDataset):
    """Endless random crops at 16 kHz, drawn from several groups of files with fixed shares."""

    def __init__(self, groups, seconds):
        self.groups, self.seconds = groups, seconds

    def __iter__(self):
        torch.set_num_threads(1)
        info = torch.utils.data.get_worker_info(); rng = random.Random(time.time_ns() + (info.id if info else 0))
        lists, shares = [g[0] for g in self.groups], [g[1] for g in self.groups]
        while True:
            path = rng.choice(rng.choices(lists, shares)[0])
            try:
                i = sf.info(path); n = int(self.seconds * i.samplerate)
                if i.frames < n:
                    continue
                w, _ = sf.read(path, start=rng.randint(0, i.frames - n), frames=n, dtype="float32", always_2d=True)
            except Exception:
                continue
            x = torchaudio.functional.resample(torch.from_numpy(w[:, 0]), i.samplerate, SR)
            yield x[: int(self.seconds * SR)]


@torch.no_grad()
def score(model, teacher, files, dev, seconds=20):
    model.eval(); cos = []
    for p in files:
        w, sr = sf.read(p, dtype="float32", always_2d=True, frames=int(seconds * sf.info(p).samplerate))
        x = torchaudio.functional.resample(torch.from_numpy(w[:, 0]).to(dev)[None], sr, SR)
        y, s = teacher_features(teacher, x)[0], model.encode(x)[0].float(); n = min(len(y), len(s))
        act = F.avg_pool1d(x.abs()[None], 320, 320)[0, 0][:n] > 0.003
        if act.sum() > 10:
            cos.append(F.cosine_similarity(s[:n], y[:n], dim=-1)[act].mean().item())
    model.train()
    return float(np.mean(cos))


ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True); ap.add_argument("--device", default="cuda:1")
ap.add_argument("--steps", type=int, default=12000); ap.add_argument("--batch", type=int, default=24)
ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--workers", type=int, default=3)
ap.add_argument("--eval-every", type=int, default=2000); ap.add_argument("--seconds", type=float, default=3.0)
a = ap.parse_args()
torch.manual_seed(0); dev = a.device
out = Path("runs/models") / a.name; out.mkdir(parents=True, exist_ok=True)
libri = [str(p) for d in ("train-clean-100", "train-clean-360") for p in Path("data/raw/libritts_r/LibriTTS_R", d).rglob("*.wav")]
vctk = [str(p) for p in Path("data/raw/vctk/wav48_silence_trimmed").rglob("*_mic1.flac")]
ears_all = sorted(Path("data/raw/ears").glob("p*/*.wav"))
ears = [str(p) for p in ears_all if p.parent.name not in HELD]
expressive = [p for p in ears if any(k in Path(p).name for k in EXPRESSIVE)]
held = [p for p in ears_all if p.parent.name in HELD]
tests = {"audiobook": sorted(str(p) for p in Path("data/raw/libritts_r/LibriTTS_R/test-clean").rglob("*.wav"))[::80][:60],
         "held speech": [str(p) for p in held if p.name.startswith(("rainbow_01_regular", "freeform_speech_01", "emo_neutral_sent"))],
         "held whisper": [str(p) for p in held if "whisper" in p.name][:15],
         "held laughter": [str(p) for p in held if "laughter" in p.name],
         "held crying": [str(p) for p in held if "crying" in p.name]}
groups = [(libri, 0.35), (vctk, 0.2), (ears, 0.3), (expressive, 0.15)]
print(f"libri {len(libri)} vctk {len(vctk)} ears {len(ears)} expressive {len(expressive)}; tests " + ", ".join(f"{k} {len(v)}" for k, v in tests.items()), flush=True)
loader = torch.utils.data.DataLoader(Mixed(groups, a.seconds), batch_size=a.batch, num_workers=a.workers, drop_last=True)
teacher = load_teacher(dev)
ck = torch.load(shared("encoder.pt"), map_location="cpu"); cfg = ck["cfg"]
model = StreamingEncoder(**cfg).to(dev); model.load_state_dict(ck["model"])
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.98), weight_decay=0.01)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / 500, 0.5 * (1 + np.cos(np.pi * min(s / a.steps, 1.0)))))
log = [{"step": 0, **{k: score(model, teacher, v, dev) for k, v in tests.items()}}]
print("== " + json.dumps(log[-1]), flush=True)
step, t0 = 0, time.time()
for wav in loader:
    wav16 = wav.to(dev); wav16 = wav16[:, : HOP * (wav16.shape[1] // HOP)]
    target = teacher_features(teacher, wav16)
    gain = torch.empty(wav16.shape[0], 1, device=dev).uniform_(-12, 3)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = model(wav16 * 10 ** (gain / 20))
    loss, cos = distill_loss(pred, target, cfg["lookahead"])
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); step += 1
    if step % 250 == 0:
        print(f"step {step} loss {loss.item():.4f} cos {cos.item():.4f} {step / (time.time() - t0):.1f} it/s", flush=True)
    if step % a.eval_every == 0 or step == a.steps:
        log.append({"step": step, **{k: score(model, teacher, v, dev) for k, v in tests.items()}})
        print("== " + json.dumps(log[-1]), flush=True)
        torch.save({"model": model.state_dict(), "cfg": cfg, "step": step}, out / "encoder.pt")
        (out / "log.json").write_text(json.dumps(log, indent=1))
    if step >= a.steps:
        break
