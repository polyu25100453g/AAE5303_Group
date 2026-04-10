#!/usr/bin/env python3
"""Build unet leaderboard JSON from evaluate_segmentation.py output (--as-percent)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "metrics_json",
        type=Path,
        help="Path to segmentation_metrics.json (with miou, dice_score, fwiou).",
    )
    p.add_argument("--group", default="Deepthinker")
    p.add_argument(
        "--repo",
        default="https://github.com/polyu25100453g/AAE5303_Group.git",
    )
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Write to file; default: print to stdout.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    m = json.loads(args.metrics_json.read_text(encoding="utf-8"))
    need = ("miou", "dice_score", "fwiou")
    for k in need:
        if k not in m:
            raise SystemExit(f"Missing key {k!r} in {args.metrics_json}")
    out = {
        "group_name": args.group,
        "project_private_repo_url": args.repo,
        "metrics": {k: m[k] for k in need},
    }
    text = json.dumps(out, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"Wrote {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
