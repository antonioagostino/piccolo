from pathlib import Path

import pytest

from src.download_pretraining_dataset import (
    build_download_plan,
    is_culturax_language_file,
    local_culturax_filename,
    normalize_dataset_name,
    select_culturax_files,
)


def test_normalize_dataset_name_supports_culturax_only():
    assert normalize_dataset_name("culturax") == "CulturaX"

    with pytest.raises(ValueError):
        normalize_dataset_name("other")


def test_culturax_file_selection_filters_language_parquets():
    repo_files = [
        "it/0000.parquet",
        "it/0001.parquet",
        "en/0000.parquet",
        "README.md",
        "metadata.json",
        "it-notes.txt",
    ]

    selected_files = select_culturax_files(repo_files, language="it", max_files=1)

    assert selected_files == ["it/0000.parquet"]


def test_culturax_language_file_matches_flat_and_nested_layouts():
    assert is_culturax_language_file("it/0000.parquet", "it")
    assert is_culturax_language_file("it_part_00000.parquet", "it")
    assert not is_culturax_language_file("en/0000.parquet", "it")


def test_local_culturax_filename_flattens_nested_source_path():
    assert local_culturax_filename("it/0000.parquet", "it") == "it_0000.parquet"
    assert local_culturax_filename("it_part_00000.parquet", "it") == "it_part_00000.parquet"


def test_build_download_plan_targets_dataset_folder():
    plan = build_download_plan(
        dataset_name="CulturaX",
        output_dir=Path("data/raw_text"),
        language="it",
        repo_id="uonlp/CulturaX",
        repo_files=["it/0000.parquet", "en/0000.parquet"],
    )

    assert plan.dataset_name == "CulturaX"
    assert plan.source_files == ["it/0000.parquet"]
    assert plan.output_dir == Path("data/raw_text")
