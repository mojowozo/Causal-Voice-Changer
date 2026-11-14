"""Distil ContentVec into the causal streaming encoder.

    uv run python -m causalvc.encoder.train --name enc-la2 --lookahead 2 --device cuda:0

The teacher sees the clean crop; the student sees the same crop (with a random
gain) and must reproduce the teacher's features causally.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

from causalvc.augment import degrade
from causalvc.encoder.model import HOP, SR, StreamingEncoder

TEACHER = "/mnt/ssd/Applio-3.7.1/rvc/models/embedders/contentvec"


class Crops(torch.utils.data.Dataset):
    """Random fixed-length crops at the files' native rate (resampled on GPU)."""

    def __init__(self, files: list[str], seconds: float, native_sr: int):
        self.files, self.n = files, int(seconds * native_sr)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int) -> torch.Tensor:
        for _ in range(20):
            path = self.files[i]
            frames = sf.info(path).frames
            if frames >= self.n:
                start = random.randint(0, frames - self.n)
                wav, _ = sf.read(path, start=start, frames=self.n, dtype="float32", always_2d=True)
                return torch.from_numpy(wav[:, 0])
            i = random.randrange(len(self.files))
        raise RuntimeError("no file long enough")


def load_teacher(device: str):
    from transformers import HubertModel

    return HubertModel.from_pretrained(TEACHER).to(device).eval().requires_grad_(False)


@torch.no_grad()
def teacher_features(teacher, wav16: torch.Tensor) -> torch.Tensor:
    return teacher(wav16).last_hidden_state


def distill_loss(pred: torch.Tensor, target: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """pred step t is trained on target frame t - k."""
    n = min(pred.shape[1] - k, target.shape[1])
    p, y = pred[:, k : k + n].float(), target[:, :n].float()
    cos = F.cosine_similarity(p, y, dim=-1).mean()
    return F.l1_loss(p, y) + (1 - cos), cos


@torch.no_grad()
def evaluate(model, teacher, files: list[str], device: str, degraded: bool = False) -> float:
    model.eval()
    cos = []
    torch.manual_seed(1)  # the same degradations at every evaluation
    for path in files:
        wav, sr = sf.read(path, dtype="float32", always_2d=True)
        x = torchaudio.functional.resample(torch.from_numpy(wav[:, 0]).to(device)[None], sr, SR)
        y = teacher_features(teacher, x)
        p = model.encode(degrade(x) if degraded else x)
        n = min(p.shape[1], y.shape[1])
        cos.append(F.cosine_similarity(p[:, :n], y[:, :n], dim=-1).mean().item())
    model.train()
    return float(np.mean(cos))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--train", nargs="+", default=["data/raw/libritts_r/LibriTTS_R/train-clean-100"])
    ap.add_argument("--valid", default="data/raw/libritts_r/LibriTTS_R/test-clean")
    ap.add_argument("--out", default="runs/models")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--lookahead", type=int, default=2)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--eval-every", type=int, default=2500)
    ap.add_argument("--augment", action="store_true", help="degrade the student's input like a real microphone")
    ap.add_argument("--init", help="encoder.pt to start from (weights only)")
    ap.add_argument("--resume", action="store_true", help="continue from the last saved weights (optimizer state is rebuilt)")
    args = ap.parse_args()

    torch.manual_seed(0)
    random.seed(0)
    out = Path(args.out) / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    files = sorted(str(p) for d in args.train for p in Path(d).rglob("*.wav"))
    valid = sorted(str(p) for p in Path(args.valid).rglob("*.wav"))
    random.Random(0).shuffle(valid)
    valid = valid[:200]
    native = sf.info(files[0]).samplerate
    loader = torch.utils.data.DataLoader(
        Crops(files, args.seconds, native), batch_size=args.batch, shuffle=True, drop_last=True,
        num_workers=args.workers, persistent_workers=True,
    )
    dev = args.device
    teacher = load_teacher(dev)
    cfg = {"dim": args.dim, "layers": args.layers, "lookahead": args.lookahead}
    model = StreamingEncoder(**cfg).to(dev)
    print(f"{len(files)} files, {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)
    warm = 2000
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, 0.5 * (1 + np.cos(np.pi * min(s / args.steps, 1.0))))
    )
    resample = torchaudio.transforms.Resample(native, SR).to(dev)
    step, t0, log = 0, time.time(), []
    if args.resume and (out / "encoder.pt").exists():
        ck = torch.load(out / "encoder.pt", map_location=dev)
        model.load_state_dict(ck["model"])
        step, log = ck["step"], json.loads((out / "log.json").read_text())
        for _ in range(step):
            sched.step()
        print(f"resumed at step {step}", flush=True)
    elif args.init:
        model.load_state_dict(torch.load(args.init, map_location=dev)["model"])
        print(f"initialised from {args.init}", flush=True)
    start = step
    while step < args.steps:
        for wav in loader:
            wav16 = resample(wav.to(dev, non_blocking=True))
            wav16 = wav16[:, : HOP * (wav16.shape[1] // HOP)]
            target = teacher_features(teacher, wav16)
            gain = torch.empty(wav16.shape[0], 1, device=dev).uniform_(-12, 3)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model((degrade(wav16) if args.augment else wav16) * 10 ** (gain / 20))
            loss, cos = distill_loss(pred, target, args.lookahead)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 250 == 0:
                print(f"step {step} loss {loss.item():.4f} cos {cos.item():.4f} "
                      f"{(step - start) / (time.time() - t0):.1f} it/s", flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                v = evaluate(model, teacher, valid, dev)
                vd = evaluate(model, teacher, valid, dev, degraded=True)
                log.append({"step": step, "valid_cos": v, "valid_cos_degraded": vd})
                print(f"== step {step} valid cos {v:.4f} degraded {vd:.4f}", flush=True)
                torch.save({"model": model.state_dict(), "cfg": cfg, "step": step}, out / "encoder.pt")
                (out / "log.json").write_text(json.dumps(log, indent=1))
            if step >= args.steps:
                break


if __name__ == "__main__":
    main()
