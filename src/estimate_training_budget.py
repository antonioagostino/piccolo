from __future__ import annotations

import argparse
from pathlib import Path

from src.train import load_training_config
from src.training_budget import print_token_budget_estimate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate token counts and max_iterations for one training pass."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training.yaml"),
        help="Path to the YAML training config.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print_token_budget_estimate(load_training_config(args.config))
