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
    language: str
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


def normalize_dataset_name(dataset_name: str) -> str:
    """
    Normalise a dataset name string to its canonical form.

    Args:
        dataset_name (str): Raw dataset name (case-insensitive).

    Returns:
        str: Canonical dataset name (e.g. ``"CulturaX"``).

    Raises:
        ValueError: If the dataset name is not supported.
    """
    normalized_name = dataset_name.strip().lower()
    if normalized_name == "culturax":
        return "CulturaX"
    raise ValueError(f"Unsupported pre-training dataset: {dataset_name}")


def is_culturax_language_file(file_path: str, language: str) -> bool:
    """
    Check whether a repository file path belongs to a given CulturaX language.

    A file matches if it is a Parquet file and either its path contains the
    language code as a path component or its filename starts with the language
    code followed by ``_`` or ``-``.

    Args:
        file_path (str): File path as returned by the Hugging Face repo
            listing.
        language (str): Language code to match (e.g. ``"it"``).

    Returns:
        bool: True if the file belongs to the requested language.
    """
    if not file_path.endswith(".parquet"):
        return False

    path = Path(file_path)
    if language in path.parts:
        return True

    return path.name.startswith(f"{language}_") or path.name.startswith(f"{language}-")


def local_culturax_filename(source_path: str, language: str) -> str:
    """
    Derive a local filename for a downloaded CulturaX file.

    If the source filename already starts with the language prefix, it is
    used as-is. Otherwise, path separators are replaced with underscores to
    produce a flat filename.

    Args:
        source_path (str): Source file path from the Hugging Face repository.
        language (str): Language code used to detect the filename prefix.

    Returns:
        str: Local filename to use when saving the downloaded file.
    """
    path = Path(source_path)
    if path.name.startswith(f"{language}_") or path.name.startswith(f"{language}-"):
        return path.name

    return source_path.replace("/", "_")


def select_culturax_files(repo_files: list[str],
                          language: str,
                          max_files: int | None = None) -> list[str]:
    """
    Filter and sort CulturaX repository files for a given language.

    Args:
        repo_files (list[str]): All file paths in the Hugging Face repository.
        language (str): Language code to filter by.
        max_files (int | None): Optional cap on the number of files returned.
            Pass None to return all matching files.

    Returns:
        list[str]: Sorted list of matching file paths, truncated to max_files.
    """
    selected_files = sorted(
        file_path
        for file_path in repo_files
        if is_culturax_language_file(file_path, language)
    )
    if max_files is not None:
        selected_files = selected_files[:max_files]

    return selected_files


def build_download_plan(dataset_name: str,
                        output_dir: Path,
                        language: str,
                        repo_id: str,
                        repo_files: list[str],
                        max_files: int | None = None) -> DatasetDownloadPlan:
    """
    Build a DatasetDownloadPlan for the requested dataset and language.

    Args:
        dataset_name (str): Human-readable dataset name (normalised
            internally).
        output_dir (Path): Root directory where the dataset will be saved.
        language (str): Language code to download.
        repo_id (str): Hugging Face repository ID.
        repo_files (list[str]): File paths listed from the repository.
        max_files (int | None): Optional cap on the number of files to
            include.

    Returns:
        DatasetDownloadPlan: The constructed download plan.

    Raises:
        ValueError: If the dataset name is not supported.
    """
    normalized_dataset_name = normalize_dataset_name(dataset_name)
    if normalized_dataset_name != "CulturaX":
        raise ValueError(f"Unsupported pre-training dataset: {dataset_name}")

    return DatasetDownloadPlan(
        dataset_name=normalized_dataset_name,
        repo_id=repo_id,
        language=language,
        output_dir=output_dir,
        source_files=select_culturax_files(repo_files, language, max_files),
    )


def download_culturax(plan: DatasetDownloadPlan,
                      revision: str,
                      token: str | bool | None) -> list[DownloadedDatasetFile]:
    """
    Execute a CulturaX download plan, copying files to the output directory.

    Args:
        plan (DatasetDownloadPlan): The download plan produced by
            build_download_plan.
        revision (str): Repository revision (branch, tag, or commit hash).
        token (str | bool | None): Hugging Face authentication token, or
            True to use the cached login token, or None/False for anonymous
            access.

    Returns:
        list[DownloadedDatasetFile]: Records of each downloaded file with its
            source path and local destination path.
    """
    huggingface_hub = get_huggingface_hub()
    dataset_output_dir = plan.output_dir / plan.dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    downloaded_files: list[DownloadedDatasetFile] = []
    for source_file in tqdm(plan.source_files, desc=f"downloading {plan.dataset_name}", unit="file"):
        cached_file_path = huggingface_hub.hf_hub_download(
            repo_id=plan.repo_id,
            filename=source_file,
            repo_type="dataset",
            revision=revision,
            token=token,
        )
        destination_path = dataset_output_dir / local_culturax_filename(source_file, plan.language)
        shutil.copy2(cached_file_path, destination_path)
        downloaded_files.append(
            DownloadedDatasetFile(
                source_path=source_file,
                destination_path=destination_path,
            )
        )

    return downloaded_files


def download_pretraining_dataset(dataset_name: str,
                                 output_dir: Path,
                                 language: str,
                                 repo_id: str,
                                 revision: str = "main",
                                 token: str | bool | None = None,
                                 max_files: int | None = None,
                                 dry_run: bool = False) -> list[DownloadedDatasetFile]:
    """
    Download a pre-training dataset from the Hugging Face Hub.

    Lists repository files, builds a download plan, and downloads only the
    files that match the requested language. Prints a summary before
    downloading.

    Args:
        dataset_name (str): Name of the dataset to download (e.g.
            ``"CulturaX"``).
        output_dir (Path): Root directory where downloaded files are saved.
        language (str): Language code to download.
        repo_id (str): Hugging Face dataset repository ID.
        revision (str): Repository revision. Defaults to ``"main"``.
        token (str | bool | None): Hugging Face authentication token.
        max_files (int | None): Optional cap on number of files to download.
        dry_run (bool): If True, print matching files without downloading
            anything.

    Returns:
        list[DownloadedDatasetFile]: Downloaded file records, or an empty
            list for dry runs.

    Raises:
        ValueError: If no matching files are found in the repository.
    """
    huggingface_hub = get_huggingface_hub()
    repo_files = huggingface_hub.list_repo_files(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        token=token,
    )
    plan = build_download_plan(
        dataset_name=dataset_name,
        output_dir=output_dir,
        language=language,
        repo_id=repo_id,
        repo_files=repo_files,
        max_files=max_files,
    )
    if not plan.source_files:
        raise ValueError(
            f"No parquet files found for {plan.dataset_name} language '{language}' "
            f"in Hugging Face repo {repo_id}."
        )

    print(f"dataset: {plan.dataset_name}")
    print(f"repo_id: {plan.repo_id}")
    print(f"language: {plan.language}")
    print(f"files: {len(plan.source_files)}")
    print(f"destination: {plan.output_dir / plan.dataset_name}")

    if dry_run:
        for source_file in plan.source_files:
            print(source_file)
        return []

    return download_culturax(plan, revision=revision, token=token)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the dataset download entry point.

    Returns:
        argparse.Namespace: Parsed arguments controlling which dataset,
            language, and files to download.
    """
    parser = argparse.ArgumentParser(description="Download pre-training datasets.")
    parser.add_argument(
        "--dataset",
        choices=["CulturaX"],
        default="CulturaX",
        help="Pre-training dataset to download.",
    )
    parser.add_argument(
        "--language",
        default="it",
        help="Language code to download. For CulturaX, use the dataset language config code.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw_text"),
        help="Directory where dataset folders are written.",
    )
    parser.add_argument(
        "--repo-id",
        default="uonlp/CulturaX",
        help="Hugging Face dataset repository id.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Hugging Face repository revision.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token. If omitted, the local HF login/token cache is used.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap for smoke downloads.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List matching files without downloading them.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_pretraining_dataset(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        language=args.language,
        repo_id=args.repo_id,
        revision=args.revision,
        token=args.token,
        max_files=args.max_files,
        dry_run=args.dry_run,
    )
