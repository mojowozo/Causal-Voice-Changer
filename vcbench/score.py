"""Scoring stage: turn the wavs written by `vcbench.convert` into metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from vcbench import audio
from vcbench.metrics import signal

SR = 16000

COLUMNS = {
    "system": "system",
    "config": "config",
    "n": "n",
    "sim_target": "sim→target ↑",
    "sim_source": "sim→source ↓",
    "wer": "WER % ↓",
    "per": "phone err % ↓",
    "utmos": "UTMOS ↑",
    "f0_pcc": "F0 corr ↑",
    "vuv_err": "V/UV err % ↓",
    "block_ripple_db": "block ripple dB ↓",
    "drift_db": "vs offline dB ↓",
    "delay_ms": "delay ms",
    "compute_p50_ms": "compute p50 ms",
    "compute_p99_ms": "compute p99 ms",
    "rtf": "RTF",
    "latency_ms": "total latency ms ↓",
    "realtime_ok": "realtime",
}


class Scorer:
    def __init__(self, manifest: dict, device: str):
        from vcbench.metrics.models import MosPredictor, PhoneRecognizer, PitchTracker, SpeakerEmbedder, Transcriber

        self.manifest = manifest
        self.sources = {s["id"]: s for s in manifest["sources"]}
        self.spk = SpeakerEmbedder(device)
        self.asr = Transcriber(device)
        self.pitch = PitchTracker(device)
        self.mos = MosPredictor(device)
        self.phone = PhoneRecognizer(device)
        self._src_phones: dict[str, str] = {}
        self._centroids: dict[str, np.ndarray] = {}
        self._src_f0: dict[str, tuple] = {}

    def centroid(self, key: str, wavs: list[str]) -> np.ndarray:
        if key not in self._centroids:
            self._centroids[key] = self.spk.centroid([audio.load(w, SR)[0] for w in wavs])
        return self._centroids[key]

    def source_f0(self, source_id: str) -> tuple:
        if source_id not in self._src_f0:
            self._src_f0[source_id] = self.pitch.track(audio.load(self.sources[source_id]["wav"], SR)[0])
        return self._src_f0[source_id]

    def source_phones(self, source_id: str) -> str:
        if source_id not in self._src_phones:
            self._src_phones[source_id] = self.phone.phones(audio.load(self.sources[source_id]["wav"], SR)[0])
        return self._src_phones[source_id]

    @staticmethod
    def offline_dir(cfg_dir: Path) -> Path:
        """Whole-file renders to compare against: the run's own, or those of the
        run named in `offline_ref.txt` (same voices, different streaming engine)."""
        ref = cfg_dir.parent / "offline_ref.txt"
        if ref.exists():
            return cfg_dir.parent.parent / ref.read_text().strip() / "offline"
        return cfg_dir.parent / "offline"

    def score_config(self, cfg_dir: Path) -> list[dict]:
        meta = json.loads((cfg_dir / "meta.json").read_text())
        io_sr, block_size = meta["io_sr"], meta["block_size"]
        rows, aligned16 = [], []
        for item in tqdm(meta["items"], desc=f"{cfg_dir.parent.name}/{cfg_dir.name}", leave=False, disable=not sys.stderr.isatty()):
            src = self.sources[item["source_id"]]
            src_io, _ = audio.load(src["wav"], io_sr)
            out_io, _ = audio.load(item["wav"], io_sr)
            delay = signal.estimate_delay_ms(src_io, out_io, io_sr)
            out_al = signal.align(out_io, delay, io_sr, len(src_io))
            out16 = audio.resample(out_al, io_sr, SR)
            aligned16.append(out16)

            emb = self.spk.embed(out16)
            tgt = self.centroid("t:" + item["target"], self.manifest["targets"][item["target"]]["ref"])
            srcc = self.centroid("s:" + src["speaker"], self.manifest["source_refs"][src["speaker"]])
            pcc, vuv = self.pitch_agreement(item["source_id"], out16)
            row = {
                "source_id": item["source_id"],
                "target": item["target"],
                "sim_target": float(emb @ tgt),
                "sim_source": float(emb @ srcc),
                "utmos": self.mos.score(out16),
                "f0_pcc": pcc,
                "vuv_err": vuv,
                "delay_ms": delay,
                "ref_text": self.asr.normalize(src["text"]),
                "ref_phones": self.source_phones(item["source_id"]),
                "hyp_phones": self.phone.phones(out16),
            }
            if block_size:
                row["ripple_var"], row["ripple_floor"] = signal.block_ripple(out_io, io_sr, block_size)
                row["block_times_ms"] = item["block_times_ms"]
                offline = self.offline_dir(cfg_dir) / Path(item["wav"]).name
                if offline.exists():
                    off_io, _ = audio.load(offline, io_sr)
                    off_al = signal.align(off_io, signal.estimate_delay_ms(src_io, off_io, io_sr), io_sr, len(src_io))
                    row["drift_db"] = signal.mel_distance_db(out_al, off_al, io_sr)
            rows.append(row)
        for row, hyp in zip(rows, self.asr.transcribe(aligned16)):
            row["hyp_text"] = hyp
        return rows

    def pitch_agreement(self, source_id: str, out16: np.ndarray) -> tuple[float, float]:
        from vcbench.metrics.models import f0_agreement

        return f0_agreement(self.source_f0(source_id), self.pitch.track(out16))


def summarize(cfg_dir: Path, rows: list[dict]) -> dict:
    import jiwer

    meta = json.loads((cfg_dir / "meta.json").read_text())
    df = pd.DataFrame(rows)
    out = {
        "system": cfg_dir.parent.name,
        "config": cfg_dir.name,
        "n": len(df),
        "sim_target": df.sim_target.mean(),
        "sim_source": df.sim_source.mean(),
        "wer": 100 * jiwer.wer(list(df.ref_text), list(df.hyp_text)),
        # Phone errors against the recogniser's own reading of the source audio.
        "per": 100 * jiwer.wer(list(df.ref_phones), list(df.hyp_phones)) if "ref_phones" in df else np.nan,
        "utmos": df.utmos.mean(),
        "f0_pcc": df.f0_pcc.mean(),
        "vuv_err": 100 * df.vuv_err.mean(),
        "delay_ms": df.delay_ms.median(),
    }
    if meta["block_size"]:
        times = np.concatenate([np.asarray(t) for t in df.block_times_ms])
        block_ms = meta["block_ms"]
        p99 = float(np.percentile(times, 99))
        out |= {
            "block_ripple_db": float(np.sqrt(max(df.ripple_var.mean() - df.ripple_floor.mean(), 0.0))),
            "drift_db": df.drift_db.mean() if "drift_db" in df else np.nan,
            "compute_p50_ms": float(np.percentile(times, 50)),
            "compute_p99_ms": p99,
            "rtf": float(times.mean() / block_ms),
            # A sample waits for its block to fill, then for the system's own
            # delay, then for the block to be computed.
            "latency_ms": block_ms + out["delay_ms"] + p99,
            "realtime_ok": bool(p99 < block_ms),
        }
    return out


def report(runs_dir: Path) -> pd.DataFrame:
    rows = []
    for scores in sorted(runs_dir.glob("*/*/scores.json")):
        rows.append(summarize(scores.parent, json.loads(scores.read_text())))
    if not rows:
        raise SystemExit(f"no scores.json under {runs_dir}; run `vcbench score` first")
    df = pd.DataFrame(rows).reindex(columns=list(COLUMNS))
    order = df.config.map(lambda c: -1.0 if c == "offline" else float(c[5:-2]))
    df = df.assign(_o=order).sort_values(["system", "_o"]).drop(columns="_o")
    df.to_csv(runs_dir / "summary.csv", index=False)
    return df


def format_report(df: pd.DataFrame) -> str:
    return df.rename(columns=COLUMNS).to_markdown(index=False, floatfmt=".3g", missingval="–")


def main(args: argparse.Namespace) -> None:
    runs_dir = Path(args.runs_dir)
    manifest = json.loads(Path(args.manifest).read_text())
    todo = []
    for meta in sorted(runs_dir.glob("*/*/meta.json")):
        if args.run and meta.parent.parent.name not in args.run:
            continue
        scores = meta.parent / "scores.json"
        if args.force or not scores.exists() or scores.stat().st_mtime < meta.stat().st_mtime:
            todo.append(meta.parent)
    if todo:
        scorer = Scorer(manifest, args.device)
        for cfg_dir in todo:
            (cfg_dir / "scores.json").write_text(json.dumps(scorer.score_config(cfg_dir)))
    print(format_report(report(runs_dir)))
