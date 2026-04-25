from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union
import random
import pandas as pd
import torch
from src.tokenizer import Tokenizer, TiktokenTokenizer


@dataclass(frozen=True)
class SplitTokenCounts:
    train_tokens: int
    val_tokens: int


def load_text_from_parquet_file(dataset_file_path: Union[str, Path],
                                raw_texts_column_name: str) -> list[str]:
    """
    A utility function for loading raw texts from a Parquet file.
    Args:
        dataset_file_path: the Parquet file's path.
        raw_texts_column_name: the file's column containing the raw texts.
    """
    return pd.read_parquet(dataset_file_path)[raw_texts_column_name].to_list()


PRE_TRAINING_DATASETS_EXTENSIONS = {
    "CulturaX": ".parquet"
}
PRE_TRAINING_DATASETS_FN = {
    "CulturaX": load_text_from_parquet_file
}
PRE_TRAINING_DATASETS_ARGS = {
    "CulturaX": ["text"]
}


def get_pre_training_dataset_files(data_path: Union[str, Path],
                                   rng: Any) -> Generator[Path, None, None]:
    pre_training_datasets_dirs = [ds_dir for ds_dir in Path(data_path).iterdir() if ds_dir.is_dir()]

    # Shuffle to avoid bias coming from files and dirs ordering
    rng.shuffle(pre_training_datasets_dirs)

    for pre_training_dataset_dir in pre_training_datasets_dirs:
        pre_training_files = [file for file in pre_training_dataset_dir.iterdir() if file.is_file()]
        # Shuffle to avoid bias coming from files and dirs ordering
        rng.shuffle(pre_training_files)
        for file in pre_training_files:
            yield file


def validate_pre_training_file(file: Path) -> None:
    dataset_name = file.parent.stem
    expected_extension = PRE_TRAINING_DATASETS_EXTENSIONS[dataset_name]
    if file.suffix != expected_extension:
        raise ValueError(f"Pretraining file's extension under directory '{dataset_name}' do not match the\
                        correct extension mapping: {expected_extension}")


def load_pre_training_raw_texts(file: Path) -> list[str]:
    dataset_name = file.parent.stem
    return PRE_TRAINING_DATASETS_FN[dataset_name](
        file,
        *PRE_TRAINING_DATASETS_ARGS[dataset_name]
    )


def count_pre_training_tokens(data_path: Union[str, Path],
                              train_split_size: float,
                              tokenizer: Tokenizer,
                              random_seed: int | None = None) -> SplitTokenCounts:
    assert Path(data_path).exists() and Path(data_path).is_dir(), \
        f"{data_path} is not a valid directory"
    assert train_split_size > 0 and train_split_size <= 1, \
        "The train split size must be greater that 0 and greater or equal to 1"

    rng = random.Random(random_seed) if random_seed is not None else random
    train_tokens_count = 0
    val_tokens_count = 0

    for file in get_pre_training_dataset_files(data_path, rng):
        validate_pre_training_file(file)
        raw_texts = load_pre_training_raw_texts(file)

        # Shuffle to mirror the train/val split used by PreTrainingDataset.
        rng.shuffle(raw_texts)
        split_idx = int(len(raw_texts) * train_split_size)

        for text in raw_texts[:split_idx]:
            train_tokens_count += len(tokenizer.encode(text)) + 1

        for text in raw_texts[split_idx:]:
            val_tokens_count += len(tokenizer.encode(text)) + 1

    return SplitTokenCounts(train_tokens=train_tokens_count,
                            val_tokens=val_tokens_count)
    
                    
class PreTrainingDataset:
    def __init__(self,
                 data_path: str,
                 sequence_length: int,
                 train_split_size: float,
                 batch_size: int,
                 tokenizer: Tokenizer,
                 device: torch.device,
                 random_seed: int | None = None) -> None:
        """
        A simple class for managing a pre-training dataset for LLMs pre-training.
        Given a data_path, it iterates through its children
        directories and checks datasets' files matching the mapped format and
        directory names.
        Args:
            data_path (str): the directory containing all the datasets the user
                wants to use for LLMs pre-training, as subfolders.
            sequence_length (int): the desired length of each sequence sampled from the
                pre-training dataset.
            train_split_size (float): a float between 0 and 1. The remaining will be used
                for the validation split.
            batch_size (int): number of sequences contained in each batch.
            tokenizer (Tokenizer): tokenizer used to tokenize raw text sequences.
            device (torch.device): select between 'cpu', 'cuda', 'mps', and so on.
        """
        assert Path(data_path).exists() and Path(data_path).is_dir(), \
            f"{data_path} is not a valid directory"
        assert train_split_size > 0 and train_split_size <= 1, \
            "The train split size must be greater that 0 and greater or equal to 1"

        self.data_path = data_path
        self.sequence_length = sequence_length
        self.train_split_size = train_split_size
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.device = device
        self.__rng = random.Random(random_seed) if random_seed is not None else random
        
        self.tokens_buffer: dict[str, list[int]] = {
            "train": [],
            "val": []
        }
        self.training_data_finished = False
        self.__tokens_generator = self.__get_tokenized_splits()

    def __get_tokenized_splits(self) -> Generator[tuple[list[int], list[int]], None, None]:
        """
        A generator function that iterates through the sub-directories 
        of the given main directory. For every subdir and files inside the
        subdirs, it checks the files' extension and the utility function
        for managing these files, passing the mapped parameters. For
        extending the LLMs pre-training dataset, update the 
        pre_training_datasets_extensions, pre_training_datasets_fn, and
        pre_training_datasets_args dictionaries.
        The function yields the tokenized train and val corpora.
        """
        for file in get_pre_training_dataset_files(self.data_path, self.__rng):
            validate_pre_training_file(file)
            raw_texts = load_pre_training_raw_texts(file)

            # Shuffle to avoid bias coming from files and dirs ordering
            self.__rng.shuffle(raw_texts)

            tokenized_texts = []
            for text in raw_texts:
                tokenized_texts.append(self.tokenizer.encode(text))

            split_idx = int(len(tokenized_texts) * self.train_split_size)
            train_tokenized: list[list[int]] = tokenized_texts[:split_idx]
            val_tokenized: list[list[int]] = tokenized_texts[split_idx:]

            # We do not need raw texts anymore
            del raw_texts

            train_tokenized_corpus = []
            val_tokenized_corpus = []
            for train_tokens in train_tokenized:
                train_tokens.append(self.tokenizer.get_end_token())
                train_tokenized_corpus += train_tokens

            for val_tokens in val_tokenized:
                val_tokens.append(self.tokenizer.get_end_token())
                val_tokenized_corpus += val_tokens

            yield train_tokenized_corpus, val_tokenized_corpus

    def __check_buffer_size(self,
                            buffer_type: str,
                            required_tokens: int | None = None) -> bool:
        """
        A simple utility function for checking if the train or validation token buffers
        are great enough to contruct a batch of token sequences for the next training 
        iteration.
        Args:
            buffer_type (str): 'train' or 'val' split.
        """
        assert buffer_type in ["train", "val"], f"Buffer type must be equal to 'train' or 'val'"
        if required_tokens is None:
            required_tokens = self.batch_size + self.sequence_length
        return len(self.tokens_buffer[buffer_type]) >= required_tokens
    
    def __fill_tokens_buffer(self,
                             split: str,
                             required_tokens: int | None = None) -> None:
        while not self.__check_buffer_size(split, required_tokens):
            try:
                train_buffer, val_buffer = next(self.__tokens_generator)
                self.tokens_buffer["train"] += train_buffer
                self.tokens_buffer["val"] += val_buffer
            except StopIteration:
                self.training_data_finished = True
                return


    def get_batch(self,
                  split: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        A fuction that prepares the batch of token sequences pulling them from the
        training or validation token buffers
        Args:
            split (str): 'train' or 'val' split.
        Return:
            Tuple[torch.Tensor, torch.Tensor]: the sequence and the shifted sequence.
        """
        assert split in ["train", "val"], f"Split must be equal to 'train' or 'val'"
        self.__fill_tokens_buffer(split)
        if self.training_data_finished and not self.__check_buffer_size(split):
            raise StopIteration

        x: list[torch.Tensor] = []
        y: list[torch.Tensor] = []
        for _ in range(self.batch_size):
            sequence = self.tokens_buffer[split][:self.sequence_length]
            sequence_shifted = self.tokens_buffer[split][1:self.sequence_length + 1]
            x.append(torch.tensor(sequence, dtype=torch.int64, device=self.device))
            y.append(torch.tensor(sequence_shifted, dtype=torch.int64, device=self.device))
            self.tokens_buffer[split].pop(0)

        return torch.stack(x), torch.stack(y)

    def get_sequential_batch(self,
                             split: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prepares a non-overlapping next-token batch for language model training.
        Each returned sequence advances the split buffer by sequence_length tokens,
        so every token is used as an input at most once in one dataset pass.
        """
        assert split in ["train", "val"], f"Split must be equal to 'train' or 'val'"
        required_tokens = self.batch_size * self.sequence_length + 1
        self.__fill_tokens_buffer(split, required_tokens)
        if self.training_data_finished and not self.__check_buffer_size(split, required_tokens):
            raise StopIteration

        x: list[torch.Tensor] = []
        y: list[torch.Tensor] = []
        for _ in range(self.batch_size):
            sequence = self.tokens_buffer[split][:self.sequence_length + 1]
            x.append(torch.tensor(sequence[:-1], dtype=torch.int64, device=self.device))
            y.append(torch.tensor(sequence[1:], dtype=torch.int64, device=self.device))
            del self.tokens_buffer[split][:self.sequence_length]

        return torch.stack(x), torch.stack(y)


if __name__ == "__main__":
    pre_training_data_path = "./data/raw_text/"
    tokenizer = TiktokenTokenizer()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    pre_training_dataset = PreTrainingDataset(pre_training_data_path,
                                              8,
                                              0.9,
                                              4,
                                              tokenizer,
                                              device
                                              )
    while True:
        try:
            x, y = pre_training_dataset.get_batch("train")
            print(x)
            print(y)
        except StopIteration:
            print("Training dataset finished!")
            break

    while True:
        try:
            x, y = pre_training_dataset.get_batch("val")
            print(x)
            print(y)
        except StopIteration:
            print("Validation dataset finished!")
            break
