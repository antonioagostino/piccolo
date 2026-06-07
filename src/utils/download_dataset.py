import argparse
import random
import shutil
from pathlib import Path
from tqdm.auto import tqdm  # type: ignore[import-untyped]
import huggingface_hub

def download_alpaca_gpt4_italian(hf_token: str,
                                 max_files: int,
                                 seed: int,
                                 output_dir: str) -> None:
    """Download the Alpaca GPT4 Italian datset's files from the HuggingFace Hub"""

    repo_id = "FreedomIntelligence/alpaca-gpt4-italian"
    repo_type = "dataset"
    revision = "main"

    repo_files:list[str] = huggingface_hub.list_repo_files(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        token=hf_token,
    )

    filtered_repo_files: list[str] = list()

    for file in repo_files:
        if Path(file).suffix in {".parquet", ".json", ".jsonl"}:
            filtered_repo_files.append(file)

    random.seed(seed)
    files_to_download: list[str] = random.sample(filtered_repo_files,
                                                 max_files)
    

    dataset_output_dir = Path(output_dir) / "alpaca-gpt4-italian"
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    for dataset_file in tqdm(files_to_download, desc=f"Downloading Alpaca GPT4 Italian", unit="file"):
        cached_file_path = huggingface_hub.hf_hub_download(
            repo_id=repo_id,
            filename=dataset_file,
            repo_type=repo_type,
            revision=revision,
            token=hf_token
        )

        shutil.copy2(cached_file_path, dataset_output_dir / Path(dataset_file).name)

def download_cultura_x(hf_token: str,
                       max_files: int,
                       seed: int,
                       output_dir: str) -> None:
    """Download the CulturaX datset's files from the HuggingFace Hub"""

    repo_id = "uonlp/CulturaX"
    repo_type = "dataset"
    revision = "main"

    repo_files:list[str] = huggingface_hub.list_repo_files(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        token=hf_token,
    )

    filtered_repo_files: list[str] = list()

    for file in repo_files:
        if file.startswith("it") and Path(file).suffix == ".parquet":
            filtered_repo_files.append(file)

    random.seed(seed)
    files_to_download: list[str] = random.sample(filtered_repo_files,
                                                 max_files)
    
    dataset_output_dir = Path(output_dir) / "CulturaX"
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
    """Parse command-line arguments for downloading datasets for pretraining or SFT"""
    parser = argparse.ArgumentParser(description="Download pretraining or SFT datasets.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw_text"),
        help="Directory where dataset folders are written.",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        choices=["pretraining", "sft"],
        default="pretraining",
        help="The type of dataset to download (for pre-traning or for SFT)",
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
        help="Maximum number of files to dowload from the HuggingFace Hub's repo",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducing stochastic operations (sample randomically files from datasets' filelist)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.dataset_type == "pretraining":
        download_cultura_x(args.token,
                           args.max_files,
                           args.seed,
                           args.output_dir)
    elif args.dataset_type == "sft":
        download_alpaca_gpt4_italian(args.token,
                                     args.max_files,
                                     args.seed,
                                     args.output_dir)
    else:
        raise ValueError("Dataset type not supported!")

    