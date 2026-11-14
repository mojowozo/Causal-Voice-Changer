"""Conversion stage: run one system over the eval pairs at several block sizes.

Run this with the Python environment of the system under test, e.g.

    python -m vcbench.convert --system vcbench.systems.passthrough:Passthrough \
        --run passthrough --blocks offline,10,20,40

Outputs land in runs/<run>/<config>/ as wavs plus a meta.json with timings.
Only needs numpy and soundfile on top of the system's own dependencies.
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path

import numpy as np

from vcbench import audio
from vcbench.streaming import VCSystem, Voice, stream_convert


def load_system(spec: str, kwargs: dict) -> VCSystem:
    module, cls = spec.split(":")
    return getattr(importlib.import_module(module), cls)(**kwargs)


def config_name(block: str) -> str:
    return "offline" if block == "offline" else f"block{float(block):g}ms"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", required=True, help="module:Class of a VCSystem")
    ap.add_argument("--system-args", default="{}", help="JSON kwargs for the system constructor")
    ap.add_argument("--run", required=True, help="run name (directory under --runs-dir)")
    ap.add_argument("--manifest", default="data/eval/manifest.json")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--blocks", default="offline,10,20,40,80", help="comma list of block sizes in ms, or 'offline'")
    ap.add_argument("--limit", type=int, default=0, help="only the first N pairs per target (smoke test)")
    ap.add_argument("--targets", default="", help="comma list of target ids (default: all)")
    ap.add_argument("--build-only", action="store_true", help="train/enrol voices, convert nothing")
    args = ap.parse_args(argv)

    # Resolve paths up front: a system adapter may change the working directory.
    manifest = json.loads(Path(args.manifest).read_text())
    sources = {s["id"]: s for s in manifest["sources"]}
    run_dir = (Path(args.runs_dir) / args.run).resolve()
    system = load_system(args.system, json.loads(args.system_args))
    blocks = [b.strip() for b in args.blocks.split(",") if b.strip()]

    voices = {
        t: Voice(id=t, train_wavs=target["train"], workdir=run_dir / "voices" / t)
        for t, target in manifest["targets"].items()
        if not args.targets or t in args.targets.split(",")
    }
    for voice in voices.values():
        voice.workdir.mkdir(parents=True, exist_ok=True)
    system.build_voices(list(voices.values()))

    metas: dict[str, dict] = {}
    for target_id, voice in voices.items():
        t0 = time.time()
        system.build_voice(voice)
        print(f"[{system.name}] voice {target_id} ready ({time.time() - t0:.0f}s)", flush=True)
        if args.build_only:
            continue
        system.load_voice(voice)

        pairs = [p for p in manifest["pairs"] if p["target"] == target_id]
        if args.limit:
            pairs = pairs[: args.limit]

        for block in blocks:
            cfg = config_name(block)
            out_dir = run_dir / cfg
            if cfg not in metas and (out_dir / "meta.json").exists():
                prev = json.loads((out_dir / "meta.json").read_text())
                prev["items"] = [i for i in prev["items"] if i["target"] not in voices]
                metas[cfg] = prev
            meta = metas.setdefault(
                cfg,
                {
                    "system": system.name,
                    "system_spec": args.system,
                    "system_args": json.loads(args.system_args),
                    "io_sr": system.io_sr,
                    "block_ms": None if block == "offline" else float(block),
                    "block_size": None,
                    "reported_latency_ms": None,
                    "items": [],
                },
            )
            for pair in pairs:
                src = sources[pair["source_id"]]
                wav, _ = audio.load(src["wav"], system.io_sr)
                out_path = out_dir / f"{src['id']}__{target_id}.wav"
                item = {"source_id": src["id"], "target": target_id, "wav": str(out_path)}
                system.set_pitch_shift(pair.get("semitones", 0))
                if block == "offline":
                    t0 = time.perf_counter()
                    y = system.convert_offline(wav)
                    system.sync()
                    if y is None:
                        raise SystemExit(f"{system.name} has no offline mode; drop 'offline' from --blocks")
                    item["offline_time_ms"] = (time.perf_counter() - t0) * 1000.0
                else:
                    block_size = int(round(system.io_sr * float(block) / 1000.0))
                    meta["block_size"] = block_size
                    res = stream_convert(system, wav, block_size)
                    y = res.wav
                    item["block_times_ms"] = [round(t, 4) for t in res.block_times_ms]
                    meta["reported_latency_ms"] = system.reported_latency_ms()
                audio.save(out_path, np.asarray(y, dtype=np.float32), system.io_sr)
                meta["items"].append(item)
            (out_dir / "meta.json").write_text(json.dumps(meta))
            print(f"[{system.name}] {target_id} {cfg}: {len(pairs)} utterances", flush=True)


if __name__ == "__main__":
    main()
