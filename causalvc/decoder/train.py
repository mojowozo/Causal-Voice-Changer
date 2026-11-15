"""Train the causal decoder as a multi-speaker base.

    uv run python -m causalvc.decoder.train --name dec-s0 --encoder runs/models/enc-la2-460/encoder.pt

Speakers: VCTK and EARS at 48 kHz, plus the harness's target voices (their
training clips only), so a checkpoint can be scored with `vcbench` directly.
Content, pitch and loudness are extracted on the GPU from each crop; the decoder
reconstructs the last part of the crop from them.
"""

from __future__ import annotations

import argparse
import json
import random
import signal
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

from causalvc.decoder.disc import Discriminators, EnergyLoss, MelLoss, SpecLoss, d_loss, g_loss
from causalvc.decoder.model import FRAME, SR, CausalMel, Decoder
from causalvc.decoder.model import tone as tone_of
from causalvc.infer import CausalVC, extract, load_encoder
from causalvc.pitch import Rmvpe
from causalvc.progress import Progress


def speaker_files(args) -> dict[str, list[str]]:
    spk: dict[str, list[str]] = {}
    for d in sorted(Path(args.vctk).glob("wav48_silence_trimmed/*")):
        files = sorted(str(p) for p in d.glob("*_mic1.flac"))
        if files:
            spk[f"vctk_{d.name}"] = files
    for d in sorted(Path(args.ears).glob("p*")):
        files = sorted(str(p) for p in d.glob("*.wav"))
        if files:
            spk[f"ears_{d.name}"] = files
    for name, t in json.loads(Path(args.manifest).read_text())["targets"].items():
        spk[name] = t["train"]
    spk.update(voice_files(args))
    return spk


def voice_files(args) -> dict[str, list[str]]:
    """New voices given as NAME=folder or NAME=file."""
    found: dict[str, list[str]] = {}
    for item in args.voice or []:
        name, src = item.split("=", 1)
        src = Path(src)
        files = sorted(str(p) for p in src.rglob("*") if p.suffix.lower() in (".wav", ".flac", ".ogg", ".mp3")) if src.is_dir() else [str(src)]
        assert files, f"{name}: no audio in {src}"
        found[name] = files
    return found


class Crops(torch.utils.data.Dataset):
    """A random speaker, a random file, a random crop, resampled to 48 kHz."""

    def __init__(self, spk: dict[str, list[str]], seconds: float, steps: int, only: list[str] | None = None):
        self.names, self.spk, self.seconds, self.steps = list(spk), spk, seconds, steps
        self.pick = [self.names.index(n) for n in only] if only else list(range(len(self.names)))

    def __len__(self) -> int:
        return self.steps

    def __getitem__(self, _: int):
        n = int(self.seconds * SR)
        for _try in range(8):
            s = random.choice(self.pick)
            path = random.choice(self.spk[self.names[s]])
            info = sf.info(path)
            want = int(self.seconds * info.samplerate)
            start = random.randint(0, max(0, info.frames - want))
            wav, sr = sf.read(path, start=start, frames=want, dtype="float32", always_2d=True)
            wav = torch.from_numpy(wav[:, 0])
            if sr != SR:
                wav = torchaudio.functional.resample(wav, sr, SR)
            wav = torch.nn.functional.pad(wav, (max(0, n - len(wav)), 0))[:n]  # short clips start in silence
            if wav[-n // 3 :].pow(2).mean().sqrt() > 2e-3:  # the reconstructed part must contain sound
                break
        peak = wav.abs().max().clamp_min(1e-4)
        gain = min(10 ** (random.uniform(-12.0, 3.0) / 20), 0.99 / peak.item())
        return wav * gain, s


@torch.no_grad()
def build_bank(spk: dict[str, list[str]], enc, frames: int, dev: str, cache: Path) -> torch.Tensor:
    """[speakers, frames, C] half-precision content frames of each speaker's own audio."""
    if cache.exists():
        return torch.load(cache, map_location=dev)
    bank = []
    for i, files in enumerate(spk.values()):
        feats, n = [], 0
        for path in random.Random(0).sample(files, len(files)):
            wav, sr = sf.read(path, dtype="float32", always_2d=True)
            w16 = torchaudio.functional.resample(torch.from_numpy(wav[: 60 * sr, 0]), sr, 16000).to(dev)
            if len(w16) < 8000:
                continue
            with torch.autocast("cuda", dtype=torch.bfloat16):
                feats.append(enc.encode(w16[None]).float()[0])
            n += len(feats[-1])
            if n >= frames:
                break
        f = torch.cat(feats)
        bank.append(f.repeat(frames // len(f) + 1, 1)[:frames].half())
        if i % 20 == 0:
            print(f"bank {i}/{len(spk)}", flush=True)
    bank = torch.stack(bank)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, cache)
    return bank


@torch.no_grad()
@torch.no_grad()
def warped_content(enc, wav48: torch.Tensor, frames: int) -> torch.Tensor:
    """Content of a copy whose spectrum (formants and pitch) is stretched by a random factor
    of 0.7-1.3, put back on the original frame clock. Training on it stops the decoder
    reading the voice's tone off speaker traces left in the content features, which other
    people's speech will not carry."""
    k = random.choice([14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26])
    wav16 = torchaudio.functional.resample(wav48, 3 * k, 20)  # to 16 kHz and warped by k/20 in one step
    wav16 = torch.nn.functional.pad(wav16, (0, (-wav16.shape[-1]) % 320))
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=wav48.is_cuda):
        c = enc.encode(wav16).float()
    return torch.nn.functional.interpolate(c.transpose(1, 2), size=frames, mode="linear", align_corners=False).transpose(1, 2)


def looked_up(content: torch.Tensor, bank: torch.Tensor, s: torch.Tensor, k: int = 8) -> torch.Tensor:
    """Blend each content frame with its nearest frames from the same speaker's bank,
    at a random strength per item, so the decoder trains on what conversion feeds it."""
    b = bank[s].float()  # [B, N, C]
    sim = torch.nn.functional.normalize(content, dim=-1) @ torch.nn.functional.normalize(b, dim=-1).transpose(1, 2)
    sim = sim.masked_fill(sim > 0.995, -1.0)  # not the very same frame, when the crop is in the bank
    top, idx = sim.topk(k, dim=-1)  # [B, T, k]
    w = torch.softmax(top / 0.05, dim=-1)
    near = (torch.gather(b[:, None].expand(-1, content.shape[1], -1, -1), 2, idx[..., None].expand(-1, -1, -1, b.shape[-1])) * w[..., None]).sum(2)
    r = torch.rand(len(s), 1, 1, device=content.device)
    r = torch.where(r < 0.25, torch.zeros_like(r), 0.4 + 0.6 * (r - 0.25) / 0.75)  # a quarter untouched, the rest 0.4-1.0
    return (1 - r) * content + r * near


@torch.no_grad()
def write_samples(path: Path, out: Path, pairs: list[tuple[str, str, int]], device: str) -> None:
    vc = CausalVC(str(path), device)
    out.mkdir(parents=True, exist_ok=True)
    for wav_path, target, semitones in pairs:
        wav, sr = sf.read(wav_path, dtype="float32", always_2d=True)
        x = torchaudio.functional.resample(torch.from_numpy(wav[:, 0]), sr, SR)
        y = vc.convert(x, target, semitones)
        sf.write(str(out / f"{Path(wav_path).stem}__{target}.wav"), y.cpu().numpy(), SR, subtype="PCM_16")


def _quiet_worker(_: int) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl-C is handled by the main process, which saves first


def main(on_save=None) -> None:
    """on_save(step, path) is called after each checkpoint of a voice build with that checkpoint's own decoder file."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--vctk", default="data/raw/vctk")
    ap.add_argument("--ears", default="data/raw/ears")
    ap.add_argument("--manifest", default="data/eval/manifest.json")
    ap.add_argument("--out", default="runs/models")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=300000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=2.0, help="crop fed to the encoder")
    ap.add_argument("--segment", type=int, default=48, help="100 Hz frames reconstructed per crop")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--channels", type=int, default=384)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init", help="another run's last.pt to start from (fine-tuning); the step count restarts")
    ap.add_argument("--disc", type=int, default=1, help="1: magnitude spectrogram discriminators; 2: banded complex ones")
    ap.add_argument("--d-warmup", type=int, default=0, help="steps that train only the discriminators (new ones on a trained generator)")
    ap.add_argument("--bands", type=int, default=0, help="4: add band-summed harmonics up to Nyquist to the excitation")
    ap.add_argument("--glide", action="store_true", help="excitation pitch slides between frames instead of stepping")
    ap.add_argument("--glide-next", action="store_true", help="excitation pitch slides from each frame's value to the next one's (correctly timed)")
    ap.add_argument("--lookahead", type=int, default=0, help="100 Hz frames of future input the decoder sees")
    ap.add_argument("--mel-cond", action="store_true", help="diagnostic: also feed the target's own mel (copy-synthesis ceiling)")
    ap.add_argument("--index-bank", type=int, default=0, help="content frames kept per speaker; >0 trains on looked-up content as conversion does")
    ap.add_argument("--stft-loss", type=float, default=0.0, help="weight of a linear-frequency log-STFT loss (resolves high harmonics)")
    ap.add_argument("--exc", default="full", choices=["full", "bandlimit", "fundamental"], help="which excitation partials each decoder stage is fed")
    ap.add_argument("--smooth-in", default="", help="letters: c interpolate content, p blend pitch buckets, h average hidden frames")
    ap.add_argument("--only", nargs="+", help="train on these speakers only; the speaker table is kept")
    ap.add_argument("--perturb", type=float, default=0.0, help="share of steps whose content comes from a spectrally warped copy of the crop")
    ap.add_argument("--energy-loss", type=float, default=0.0, help="weight of a loss on each band's segment-averaged power (long-term tone)")
    ap.add_argument("--decorr", type=float, default=0.0, help="weight of a loss that makes unvoiced output depend on the random noise input (breaths, whispers, sibilants)")
    ap.add_argument("--tone", type=float, default=0.0, help="experimental: also tell the decoder each frame's brightness and vowel openness; "
                    "the value is the spread (dB) of a random per-crop offset that hides the source's own overall tone")
    ap.add_argument("--voice", nargs="+", help="NAME=folder-or-file: new voices added to the base's speaker table (use with --init and --only)")
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    dev = args.device
    out = Path(args.out) / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))

    last = out / "last.pt"
    if args.voice and (args.init or args.resume):
        # Building a voice needs only its own audio: the base's speaker list comes from the checkpoint,
        # so the datasets the base was trained on do not have to be on this machine.
        source = last if args.resume and last.exists() else args.init
        spk = {name: [] for name in torch.load(source, map_location="cpu")["speakers"]}
        spk.update(voice_files(args))
    else:
        spk = speaker_files(args)
    print(f"{len(spk)} speakers, {sum(map(len, spk.values()))} files", flush=True)
    sample_pairs = []  # a few conversions to listen to while a base trains
    if not args.voice and Path(args.manifest).is_file():
        manifest = json.loads(Path(args.manifest).read_text())
        by_id = {s["id"]: s["wav"] for s in manifest["sources"]}
        for target in list(manifest["targets"])[-1:] + list(manifest["targets"])[:1]:
            for p in [p for p in manifest["pairs"] if p["target"] == target][::23][:2]:
                sample_pairs.append((by_id[p["source_id"]], target, p["semitones"]))

    enc, rmvpe = load_encoder(args.encoder, dev), Rmvpe(dev)
    k = enc.lookahead if hasattr(enc, "lookahead") else 2
    cfg = {"channels": args.channels, "dim": args.dim, "bands": args.bands, "glide": "next" if args.glide_next else args.glide, "lookahead": args.lookahead, "mel_cond": args.mel_cond, "exc": args.exc, "smooth_in": args.smooth_in, **({"tone": 2} if args.tone else {})}
    gen, disc, mel_loss = Decoder(len(spk), **cfg).to(dev), Discriminators(args.disc).to(dev), MelLoss().to(dev)
    print(f"generator {sum(p.numel() for p in gen.parameters()) / 1e6:.1f}M params", flush=True)
    opt_g = torch.optim.AdamW(gen.parameters(), args.lr, betas=(0.8, 0.99))
    opt_d = torch.optim.AdamW(disc.parameters(), args.lr, betas=(0.8, 0.99))
    step = 0
    if args.resume and last.exists():
        ck = torch.load(last, map_location=dev)
        gen.load_state_dict(ck["generator"]), disc.load_state_dict(ck["discriminator"])
        opt_g.load_state_dict(ck["opt_g"]), opt_d.load_state_dict(ck["opt_d"])
        step = ck["step"]
        print(f"resumed at step {step}", flush=True)
    elif args.init:
        ck = torch.load(args.init, map_location=dev)
        own = gen.state_dict()  # excitation convs may have gained input channels: new ones start at zero
        for a, b in ck["generator"].items():
            if own[a].shape != b.shape:  # new inputs (excitation channels, future taps) start at zero
                own[a].zero_()
                part = tuple(slice(0, min(m, n)) for m, n in zip(own[a].shape, b.shape))  # the base may also have more (speakers)
                own[a][part] = b[part]
            else:
                own[a].copy_(b)
        gen.load_state_dict(own)
        own = disc.state_dict()  # keep the discriminators whose shape is unchanged
        keep = {a: b for a, b in ck["discriminator"].items() if a in own and own[a].shape == b.shape}
        disc.load_state_dict(keep, strict=False)
        print(f"discriminator: {len(keep)} of {len(own)} tensors loaded", flush=True)
        base_n = len(ck["speakers"])
        shared = min(base_n, len(spk))
        assert ck["speakers"][:shared] == list(spk)[:shared], "speaker table differs from the base run"
        if base_n > len(spk):  # trailing speakers of the base that this run does not have are left out
            print(f"dropped {base_n - len(spk)} speaker(s) of the base: {ck['speakers'][len(spk):]}", flush=True)
        if base_n < len(spk):  # new voices start from the average of the voices the base knows
            w = gen.speaker.weight.data
            w[base_n:] = w[:base_n].mean(0, keepdim=True)
            print(f"added {len(spk) - base_n} new voice(s): {list(spk)[base_n:]}", flush=True)
        print(f"initialised from {args.init} (step {ck['step']})", flush=True)
    decay = lambda s: 0.999996**s  # noqa: E731  (halves every ~170k steps)
    sch_g = torch.optim.lr_scheduler.LambdaLR(opt_g, decay, last_epoch=-1)
    sch_d = torch.optim.lr_scheduler.LambdaLR(opt_d, decay, last_epoch=-1)
    for _ in range(step):
        sch_g.step(), sch_d.step()

    loader = torch.utils.data.DataLoader(
        Crops(spk, args.seconds, args.steps * args.batch, args.only), batch_size=args.batch,
        num_workers=args.workers, drop_last=True, persistent_workers=True, worker_init_fn=_quiet_worker,
    )
    # Reconstruct frames that end before the encoder's zero-padded lookahead region.
    t_total = int(args.seconds * 100)
    hi = t_total - 2 * (k + 2)
    lo = hi - args.segment
    cmel = CausalMel().to(dev)
    spec_loss = SpecLoss().to(dev)
    energy_loss = EnergyLoss().to(dev)
    bank = None
    if args.index_bank:
        bank = build_bank(spk, enc, args.index_bank, dev, Path("runs/cache") / f"bank_{Path(args.encoder).parent.name}_{args.index_bank}.pt")
        print(f"bank {tuple(bank.shape)}", flush=True)
    log, shown = {}, ""
    prog = Progress(args.steps, step)
    stop = {"now": False}  # set by Ctrl-C or a terminate signal: save a checkpoint, then leave

    def _stop(*_):
        stop["now"] = True

    signal.signal(signal.SIGINT, _stop), signal.signal(signal.SIGTERM, _stop)
    stop_file = out / "stop"  # creating this file asks for the same thing; it works where signals cannot (Windows, a GUI)
    stop_file.unlink(missing_ok=True)
    for wav, s in loader:
        if step >= args.steps:
            break
        if step % 10 == 0 and stop_file.exists():
            stop["now"] = True
        wav, s = wav.to(dev, non_blocking=True), s.to(dev)
        content, f0, loud = extract(enc, rmvpe, wav)
        if args.perturb and random.random() < args.perturb:
            content = warped_content(enc, wav, content.shape[1])
        if bank is not None:
            content = looked_up(content, bank, s)
        tn = tone_of(wav) + args.tone * torch.randn(wav.shape[0], 1, 2, device=dev) if args.tone else None
        h, g = gen.features(content, f0, loud, s, cmel(wav) if args.mel_cond else None, tn)
        exc = gen.source(f0[:, : h.shape[-1]])
        fake = gen.generate(h[..., lo:hi], g, exc[..., lo * FRAME : hi * FRAME])
        real = wav[:, None, lo * FRAME : hi * FRAME]

        loss_d = d_loss(disc(real), disc(fake.detach()))
        opt_d.zero_grad(set_to_none=True)
        loss_d.backward()
        opt_d.step()

        if step < args.d_warmup:
            step += 1
            sch_g.step(), sch_d.step()
            prog.update(step, f"warmup d {loss_d.item():.3f}")
            continue
        loss_mel = mel_loss(fake, real)
        with torch.no_grad():
            real_out = disc(real)
        adv, fm = g_loss(real_out, disc(fake))
        loss_g = adv + 2.0 * fm + 45.0 * loss_mel
        if args.stft_loss:
            loss_stft = spec_loss(fake, real)
            loss_g = loss_g + args.stft_loss * loss_stft
            log["stft"] = log.get("stft", 0.0) + loss_stft.item() / 100
        if args.decorr:
            # The same frames again with fresh noise in the excitation. Real breath noise is different every
            # time; without this the decoder computes one fixed noise-like pattern from the features and
            # ignores the noise it is given. Penalise the per-frame correlation of the two outputs where unvoiced.
            again = gen.generate(h[..., lo:hi], g, gen.source(f0[:, : h.shape[-1]])[..., lo * FRAME : hi * FRAME])
            a, b = (y[:, 0].unflatten(1, (-1, FRAME)) for y in (fake, again))
            a, b = a - a.mean(-1, keepdim=True), b - b.mean(-1, keepdim=True)
            corr = (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-5)
            quiet = (f0[:, lo:hi] == 0).float()
            loss_dc = (corr.pow(2) * quiet).sum() / quiet.sum().clamp_min(1.0)
            loss_g = loss_g + args.decorr * loss_dc
            log["decorr"] = log.get("decorr", 0.0) + loss_dc.item() / 100
        if args.energy_loss:
            loss_en = energy_loss(fake, real)
            loss_g = loss_g + args.energy_loss * loss_en
            log["energy"] = log.get("energy", 0.0) + loss_en.item() / 100
        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        opt_g.step()
        sch_g.step(), sch_d.step()
        step += 1

        for key, v in (("mel", loss_mel), ("d", loss_d), ("adv", adv), ("fm", fm)):
            log[key] = log.get(key, 0.0) + v.item() / 100
        if step % 100 == 0:
            shown, log = " ".join(f"{a} {b:.3f}" for a, b in log.items()), {}
        prog.update(step, shown)
        if step % args.save_every == 0 or step == args.steps or stop["now"]:
            prog.pause()
            ck = {
                "generator": gen.state_dict(), "discriminator": disc.state_dict(),
                "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(), "step": step,
                "cfg": cfg, "speakers": list(spk), "encoder": args.encoder,  # a hint only: causalvc.voice.encoder_path falls back to the shared encoder
            }
            torch.save(ck, last)
            slim = {a: b for a, b in ck.items() if a not in ("discriminator", "opt_g", "opt_d")}
            torch.save(slim, out / "decoder.pt")
            if args.voice:  # every checkpoint of a voice is kept, so different amounts of training can be compared
                torch.save(slim, out / f"decoder-{step}s.pt")
                if on_save:
                    on_save(step, out / f"decoder-{step}s.pt")
            elif step % (4 * args.save_every) == 0:
                torch.save(slim, out / f"decoder_{step // 1000}k.pt")
            gen.eval()
            if not args.mel_cond and sample_pairs and not stop["now"]:
                write_samples(out / "decoder.pt", out / "samples" / f"step_{step // 1000:04d}k", sample_pairs, dev)
            gen.train()
            prog.resume()
            prog.note(f"saved {out / 'decoder.pt'} at step {step}")
            if stop["now"]:
                stop_file.unlink(missing_ok=True)
                prog.note(f"stopped at step {step} of {args.steps}; run the same command again to resume")
                break


if __name__ == "__main__":
    main()
