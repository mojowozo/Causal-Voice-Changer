"""vcbench command line: data -> convert -> score -> report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    # `convert` has its own parser because it also runs standalone inside
    # third-party environments (python -m vcbench.convert).
    if len(sys.argv) > 1 and sys.argv[1] == "convert":
        from vcbench import convert

        return convert.main(sys.argv[2:])

    ap = argparse.ArgumentParser(prog="vcbench", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    from vcbench import data

    data.add_args(sub.add_parser("data", help="build the evaluation manifest"))
    sub.add_parser("convert", help="run a system over the eval pairs (see python -m vcbench.convert -h)")

    sc = sub.add_parser("score", help="score converted audio and print the summary")
    sc.add_argument("--runs-dir", default="runs")
    sc.add_argument("--manifest", default="data/eval/manifest.json")
    sc.add_argument("--run", action="append", help="only these runs (repeatable)")
    sc.add_argument("--device", default="cuda:0")
    sc.add_argument("--force", action="store_true")

    rp = sub.add_parser("report", help="print the summary from existing scores")
    rp.add_argument("--runs-dir", default="runs")

    args = ap.parse_args()
    if args.cmd == "data":
        data.main(args)
    elif args.cmd == "score":
        from vcbench import score

        score.main(args)
    elif args.cmd == "report":
        from vcbench import score

        print(score.format_report(score.report(Path(args.runs_dir))))


if __name__ == "__main__":
    main()
