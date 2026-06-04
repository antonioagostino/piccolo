import argparse
import random
import shutil
from pathlib import Path
from tqdm.auto import tqdm  # type: ignore[import-untyped]
import huggingface_hub

def download_cultura_x(files_to_download: list[str],
                       output_dir: str,
                       repo_id: str,
                       repo_type: str,
                       revision: str,
                       hf_token: str) -> None:
    """Download the CulturaX datset's files from the HuggingFace Hub"""

    dataset_output_dir = Path(output_dir / "CulturaX")
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    for dataset_file in tqdm(files_to_download, desc=f"Downloading CulturaX", unit="file"):
        cached_file_path = huggingface_hub.hf_hub_download(
            repo_id=repo_id,
            filename=dataset_file,
            repo_type=repo_type,
            revision=revision,
            token=hf_token
        )

        # Remove the first part of the filename (e.g. 'it_part_00057.parquet')
        truncated_dataset_filename = dataset_file.split("/")[-1]
        shutil.copy2(cached_file_path, dataset_output_dir / truncated_dataset_filename)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for downloading pre-training dataset"""
    parser = argparse.ArgumentParser(description="Download pre-training datasets.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw_text"),
        help="Directory where dataset folders are written.",
    )
    parser.add_argument(
        "--token",
        required=True,
        help="Hugging Face token to use.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        required=True,
        help="Maximum number of files to dowload from CulturaX dataset's repo",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducing stochastic operations (sample randomically files from CulturaX filelist)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    repo_id = "uonlp/CulturaX"
    revision = "main"
    repo_type = "dataset"

    repo_files:list[str] = huggingface_hub.list_repo_files(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        token=args.token,
    )

    filtered_repo_files: list[str] = list()

    for file in repo_files:
        if file.startswith("it") and file.endswith(".parquet"):
            filtered_repo_files.append(file)

    random.seed(args.seed)
    files_to_download: list[str] = random.sample(filtered_repo_files,
                                                 args.max_files)
    
    download_cultura_x(files_to_download,
                       args.output_dir,
                       repo_id,
                       repo_type,
                       revision,
                       args.token)

    