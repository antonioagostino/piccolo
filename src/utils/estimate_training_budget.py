from __future__ import annotations

import argparse
from pathlib import Path

from src.train import load_training_config
from src.utils.training_budget import print_token_budget_estimate


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the training budget estimator.

    Returns:
        argparse.Namespace: Parsed arguments with a ``config`` path and
            optional ``gpu``, ``days``, and ``batch_size`` fields for
            Chinchilla and model analysis.
    """
    parser = argparse.ArgumentParser(
        description="Estimate token counts and max_iterations for one training pass."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training.yaml"),
        help="Path to the YAML training config.",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="GPU model for Chinchilla estimate (e.g. A100_40GB, A100_80GB).",
    )
    parser.add_argument(
        "--days",
        type=float,
        default=None,
        help="Number of rental days for Chinchilla estimate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the GPU-fitted batch size with a manually verified value.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print_token_budget_estimate(
        load_training_config(args.config),
        gpu=args.gpu,
        days=args.days,
        batch_size=args.batch_size,
    )
