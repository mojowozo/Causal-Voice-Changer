"""Build the evaluation manifest.

Targets: speakers whose audio is split into a training pool (what a system may
build its voice from) and held-out reference clips (what similarity is scored
against). Sources: other speakers' utterances with transcripts. Every source
utterance is converted to every target.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from vcbench import audio


def estimate_genders(by_spk: dict[str, list[Path]], device: str) -> dict[str, str]:
    """LibriTTS-R's test-clean archive ships no speaker table, so split speakers
    by median F0 (165 Hz). Only used to balance the selection."""
    f0 = median_f0s({spk: [str(w) for w in wavs[:: max(1, len(wavs) // 4)][:4]] for spk, wavs in by_spk.items()}, device)
    return {spk: "F" if hz >= 165.0 else "M" for spk, hz in f0.items()}


def slice_long_recording(path: Path, out_dir: Path, min_s: float = 4.0, max_s: float = 10.0) -> list[str]:
    """Cut one long recording into clips at the quietest point in each window."""
    done = sorted(out_dir.glob("*.wav"))
    if done:
        return [str(p) for p in done]
    out_dir.mkdir(parents=True, exist_ok=True)
    wav, sr = audio.load(path)
    frame = int(sr * 0.02)
    n = len(wav) // frame
    rms = np.sqrt((wav[: n * frame].reshape(n, frame) ** 2).mean(axis=1) + 1e-10)
    lo, hi = int(min_s / 0.02), int(max_s / 0.02)
    clips, start = [], 0
    while start + lo < n:
        window = rms[start + lo : min(start + hi, n)]
        end = start + lo + int(np.argmin(window))
        seg = wav[start * frame : end * frame]
        # Keep clips that are mostly speech: loud enough and not half silence.
        seg_rms = rms[start:end]
        if 20 * np.log10(seg_rms.mean()) > -40 and (seg_rms > 0.25 * np.median(rms[rms > 1e-4])).mean() > 0.6:
            out = out_dir / f"{path.stem[:24]}_{len(clips):05d}.wav"
            sf.write(str(out), seg, sr, subtype="PCM_16")
            clips.append(str(out))
        start = end
    return clips


def median_f0s(groups: dict[str, list[str]], device: str) -> dict[str, float]:
    from vcbench.metrics.models import SR, PitchTracker

    tracker = PitchTracker(device)
    out = {}
    for key, wavs in groups.items():
        voiced = []
        for w in wavs[:6]:
            f, v = tracker.track(audio.load(w, SR)[0])
            voiced.append(f[v])
        out[key] = float(np.median(np.concatenate(voiced)))
    return out


def build(args: argparse.Namespace) -> dict:
    rng = random.Random(args.seed)
    root = Path(args.libritts).resolve()
    by_spk: dict[str, list[Path]] = defaultdict(list)
    for wav in sorted(root.rglob("*.wav")):
        by_spk[wav.name.split("_")[0]].append(wav)
    if not by_spk:
        raise SystemExit(f"no wav files under {root}")
    dur = {spk: sum(audio.duration(w) for w in wavs) for spk, wavs in by_spk.items()}
    genders = estimate_genders(by_spk, args.device)

    def pick(gender: str, n: int, exclude: set[str]) -> list[str]:
        pool = [s for s in by_spk if genders.get(s) == gender and s not in exclude]
        return sorted(pool, key=lambda s: -dur[s])[:n]

    # Targets get the speakers with the most audio; sources come from the rest.
    target_ids = pick("F", args.targets_per_gender, set()) + pick("M", args.targets_per_gender, set())
    source_ids = pick("F", args.sources_per_gender, set(target_ids)) + pick(
        "M", args.sources_per_gender, set(target_ids)
    )

    def eval_clips(spk: str) -> list[Path]:
        ok = [w for w in by_spk[spk] if 3.0 <= audio.duration(w) <= 10.0 and w.with_suffix(".normalized.txt").exists()]
        rng.shuffle(ok)
        return ok

    targets = {}
    for spk in target_ids:
        refs = eval_clips(spk)[: args.refs]
        train = [w for w in by_spk[spk] if w not in refs]
        targets[f"libri{spk}"] = {
            "gender": genders[spk],
            "train": [str(w) for w in train],
            "ref": [str(w) for w in refs],
            "train_minutes": round(sum(audio.duration(w) for w in train) / 60, 1),
        }

    for spec in args.extra_target or []:
        name, path = spec.split("=", 1)
        clips = slice_long_recording(Path(path), Path(args.out).resolve().parent / "voices" / name)
        # Own generator: adding a voice must not change which libri clips are drawn.
        random.Random(f"{args.seed}:{name}").shuffle(clips)
        refs, pool = clips[: args.refs], clips[args.refs :]
        train, total = [], 0.0
        for clip in pool:
            if total >= args.extra_train_minutes * 60:
                break
            train.append(clip)
            total += audio.duration(clip)
        targets[name] = {"gender": "?", "train": train, "ref": refs, "train_minutes": round(total / 60, 1)}

    sources, source_refs = [], {}
    for spk in source_ids:
        clips = eval_clips(spk)
        for w in clips[: args.utts]:
            sources.append(
                {
                    "id": w.stem,
                    "speaker": spk,
                    "gender": genders[spk],
                    "wav": str(w),
                    "text": w.with_suffix(".normalized.txt").read_text().strip(),
                }
            )
        # Different utterances of the same speaker, so leakage is scored on
        # voice identity rather than on shared content.
        source_refs[spk] = [str(w) for w in clips[args.utts : args.utts + args.refs]]

    # Static transposition per (source speaker, target): what a user would set
    # once for their own voice. Systems that model pitch themselves may ignore it.
    f0 = median_f0s(
        {f"t:{t}": v["ref"] for t, v in targets.items()} | {f"s:{s}": v for s, v in source_refs.items()},
        args.device,
    )
    for t, v in targets.items():
        v["f0_median"] = round(f0[f"t:{t}"], 1)
    manifest = {
        "targets": targets,
        "sources": sources,
        "source_refs": source_refs,
        "source_f0_median": {s: round(f0[f"s:{s}"], 1) for s in source_refs},
        "pairs": [
            {
                "source_id": s["id"],
                "target": t,
                "semitones": int(round(12 * np.log2(f0[f"t:{t}"] / f0[f"s:{s['speaker']}"]))),
            }
            for t in targets
            for s in sources
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=1))
    return manifest


def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--libritts", default="data/raw/libritts_r", help="extracted LibriTTS-R test-clean")
    ap.add_argument("--out", default="data/eval/manifest.json")
    ap.add_argument("--targets-per-gender", type=int, default=2)
    ap.add_argument("--sources-per-gender", type=int, default=2)
    ap.add_argument("--utts", type=int, default=10, help="source utterances per source speaker")
    ap.add_argument("--refs", type=int, default=10, help="held-out reference clips per speaker")
    ap.add_argument("--extra-target", action="append", help="name=/path/to/long_recording.wav (repeatable)")
    ap.add_argument("--extra-train-minutes", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")


def main(args: argparse.Namespace) -> None:
    m = build(args)
    for name, t in m["targets"].items():
        print(
            f"target {name:>12} [{t['gender']}] train {t['train_minutes']:5.1f} min, "
            f"{len(t['ref'])} refs, median F0 {t['f0_median']:.0f} Hz"
        )
    print(f"{len(m['sources'])} source utterances from {len(m['source_refs'])} speakers -> {len(m['pairs'])} pairs")
