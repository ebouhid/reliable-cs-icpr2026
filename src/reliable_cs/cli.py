"""CLI entry point: python -m reliable_cs."""

from __future__ import annotations

import argparse
from pathlib import Path

from reliable_cs.config import RANDOM_SEED
from reliable_cs.pipeline import run_campaign


def _default_repo_root() -> Path:
    # src/reliable_cs/cli.py -> repo root
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce ICPR 2026 SVM experiments on citizen-science labels."
    )
    parser.add_argument(
        "--campaign",
        choices=["landsat", "sentinel", "all"],
        default="all",
        help="Which campaign to run (default: all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed (default: {RANDOM_SEED}).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root containing data/ (default: auto-detect).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = (args.repo_root or _default_repo_root()).resolve()

    campaigns = ["landsat", "sentinel"] if args.campaign == "all" else [args.campaign]
    for key in campaigns:
        run_campaign(key, repo_root=repo_root, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
