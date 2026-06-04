"""
Download the SFT fine-tuning dataset from the Hugging Face Hub.

Downloads ``FreedomIntelligence/alpaca-gpt4-italian`` (or any other repo
passed via --repo-id) to a local directory, filtering for Parquet data files
and skipping everything else (READMEs, scripts, etc.).

Output layout (--output-dir)
-----------------------------
  <output-dir>/<dataset-name>/
      train-*.parquet   (or whatever shards the repo contains)
      ...

Usage
-----
    python -m src.utils.download_sft_dataset
    python -m src.utils.download_sft_dataset --dry-run
    python -m src.utils.download_sft_dataset --output-dir ./data/conversational
"""
from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm  # type: ignore[import-untyped]


@dataclass(frozen=True)
class DownloadedDatasetFile:
    source_path: str
    destination_path: Path


@dataclass(frozen=True)
class DatasetDownloadPlan:
    dataset_name: str
    repo_id: str
    output_dir: Path
    source_files: list[str]


def get_huggingface_hub() -> Any:
    """
    Lazily import and return the huggingface_hub module.

    Returns:
        Any: The huggingface_hub module.

    Raises:
        RuntimeError: If huggingface_hub is not installed.
    """
    try:
        return import_module("huggingface_hub")
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download datasets. "
            "Install requirements.txt before running this script."
        ) from exc


def dataset_name_from_repo_id(repo_id: str) -> str:
    """
    Derive a human-readable directory name from a Hugging Face repo ID.

    Takes the part after the last ``/`` so that
    ``"FreedomIntelligence/alpaca-gpt4-italian"`` becomes
    ``"alpaca-gpt4-italian"``.

    Args:
        repo_id (str): Hugging Face repository ID (``"owner/name"`` format).

    Returns:
        str: The dataset name portion of the repo ID.
    """
    return repo_id.split("/")[-1]


_DATA_EXTENSIONS = {".parquet", ".json", ".jsonl"}


def is_data_file(file_path: str) -> bool:
    """
    Return True if *file_path* is a recognised data file.

    Accepted formats: Parquet, JSON, JSONL.  Everything else (README,
    .gitattributes, scripts, …) is skipped.

    Args:
        file_path (str): File path as returned by the Hugging Face repo listing.

    Returns:
        bool: True for data files.
    """
    return Path(file_path).suffix in _DATA_EXTENSIONS


def select_data_files(
    repo_files: list[str],
    max_files: int | None = None,
) -> list[str]:
    """
    Filter and sort repository files, keeping only Parquet data shards.

    Args:
        repo_files (list[str]): All file paths in the Hugging Face repository.
        max_files (int | None): Optional cap on the number of files returned.

    Returns:
        list[str]: Sorted list of matching file paths, truncated to max_files.
    """
    selected = sorted(f for f in repo_files if is_data_file(f))
    if max_files is not None:
        selected = selected[:max_files]
    return selected


def build_download_plan(
    repo_id: str,
    output_dir: Path,
    repo_files: list[str],
    max_files: int | None = None,
) -> DatasetDownloadPlan:
    """
    Build a DatasetDownloadPlan for the given repository.

    Args:
        repo_id (str): Hugging Face dataset repository ID.
        output_dir (Path): Root directory where the dataset will be saved.
        repo_files (list[str]): File paths listed from the repository.
        max_files (int | None): Optional cap on the number of files.

    Returns:
        DatasetDownloadPlan: The constructed download plan.
    """
    return DatasetDownloadPlan(
        dataset_name=dataset_name_from_repo_id(repo_id),
        repo_id=repo_id,
        output_dir=output_dir,
        source_files=select_data_files(repo_files, max_files),
    )


def execute_download_plan(
    plan: DatasetDownloadPlan,
    revision: str,
    token: str | bool | None,
) -> list[DownloadedDatasetFile]:
    """
    Execute a download plan, copying each file to the output directory.

    Args:
        plan (DatasetDownloadPlan): The plan produced by build_download_plan.
        revision (str): Repository revision (branch, tag, or commit hash).
        token (str | bool | None): Hugging Face authentication token, or
            True to use the cached login token, or None for anonymous access.

    Returns:
        list[DownloadedDatasetFile]: Records of each downloaded file with its
            source path and local destination path.
    """
    huggingface_hub = get_huggingface_hub()
    dataset_dir = plan.output_dir / plan.dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    downloaded: list[DownloadedDatasetFile] = []
    for source_file in tqdm(plan.source_files,
                            desc=f"downloading {plan.dataset_name}",
                            unit="file"):
        cached_path = huggingface_hub.hf_hub_download(
            repo_id=plan.repo_id,
            filename=source_file,
            repo_type="dataset",
            revision=revision,
            token=token,
        )
        destination = dataset_dir / Path(source_file).name
        shutil.copy2(cached_path, destination)
        downloaded.append(
            DownloadedDatasetFile(
                source_path=source_file,
                destination_path=destination,
            )
        )

    return downloaded


def download_sft_dataset(
    repo_id: str,
    output_dir: Path,
    revision: str = "main",
    token: str | bool | None = None,
    max_files: int | None = None,
    dry_run: bool = False,
) -> list[DownloadedDatasetFile]:
    """
    Download an SFT dataset from the Hugging Face Hub.

    Lists repository files, builds a download plan keeping only Parquet data
    shards, prints a summary, then downloads.

    Args:
        repo_id (str): Hugging Face dataset repository ID
            (e.g. ``"FreedomIntelligence/alpaca-gpt4-italian"``).
        output_dir (Path): Root directory where downloaded files are saved.
            Files land in ``output_dir/<dataset-name>/``.
        revision (str): Repository revision. Defaults to ``"main"``.
        token (str | bool | None): Hugging Face authentication token.
        max_files (int | None): Optional cap on number of files to download.
            Useful for smoke tests.
        dry_run (bool): If True, print the file list without downloading.

    Returns:
        list[DownloadedDatasetFile]: Downloaded file records, or an empty
            list for dry runs.

    Raises:
        ValueError: If no Parquet data files are found in the repository.
    """
    huggingface_hub = get_huggingface_hub()
    repo_files = list(huggingface_hub.list_repo_files(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        token=token,
    ))

    plan = build_download_plan(
        repo_id=repo_id,
        output_dir=output_dir,
        repo_files=repo_files,
        max_files=max_files,
    )

    if not plan.source_files:
        raise ValueError(
            f"No data files (.parquet / .json / .jsonl) found in Hugging Face "
            f"repo '{repo_id}'. Check the repo ID and revision."
        )

    dataset_dir = plan.output_dir / plan.dataset_name
    print(f"dataset : {plan.dataset_name}")
    print(f"repo_id : {plan.repo_id}")
    print(f"files   : {len(plan.source_files)}")
    print(f"dest    : {dataset_dir}")

    if dry_run:
        print("\n-- dry run, files that would be downloaded --")
        for source_file in plan.source_files:
            print(f"  {source_file}")
        return []

    return execute_download_plan(plan, revision=revision, token=token)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the SFT dataset download entry point.

    Returns:
        argparse.Namespace: Parsed arguments controlling which repo and files
            to download.
    """
    parser = argparse.ArgumentParser(
        description="Download an SFT fine-tuning dataset from Hugging Face."
    )
    parser.add_argument(
        "--repo-id",
        default="FreedomIntelligence/alpaca-gpt4-italian",
        help="Hugging Face dataset repository ID "
             "(default: FreedomIntelligence/alpaca-gpt4-italian).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/conversational"),
        help="Root directory where the dataset folder is written "
             "(default: data/conversational).",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Hugging Face repository revision (default: main).",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token. If omitted, the local HF login/token cache "
             "is used.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap on number of files to download. "
             "Useful for quick smoke tests.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List matching files without downloading anything.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_sft_dataset(
        repo_id=args.repo_id,
        output_dir=args.output_dir,
        revision=args.revision,
        token=args.token,
        max_files=args.max_files,
        dry_run=args.dry_run,
    )
