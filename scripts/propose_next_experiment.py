#!/usr/bin/env python
from __future__ import annotations

import argparse

from trec26_rag.experiment_optimizer import write_next_experiment_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the next config-only experiment proposal.")
    parser.add_argument("--base-config", default="configs/baseline_retrieval.yaml")
    parser.add_argument("--output-dir", default="configs/experiments")
    parser.add_argument("--max-runs", type=int, default=25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = write_next_experiment_config(
        base_config_path=args.base_config,
        output_dir=args.output_dir,
        max_runs=args.max_runs,
    )
    print(f"Proposed experiment config: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
