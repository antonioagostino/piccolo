from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union
import random
import numpy as np
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
    Load raw texts from a single Parquet file.

    Args:
        dataset_file_path (str | Path): Path to the Parquet file.
        raw_texts_column_name (str): Name of the column containing raw text.

    Returns:
        list[str]: List of raw text strings from the specified column.
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
    """
    Yield pre-training data files from all dataset sub-directories.

    Iterates over immediate sub-directories of data_path, shuffles both
    directory and file order to avoid positional bias, then yields each
    file path one by one.

    Args:
        data_path (str | Path): Root directory containing one sub-directory
            per dataset (e.g. ``CulturaX/``).
        rng (Any): A random object with a ``shuffle`` method (e.g.
            ``random.Random``).

    Yields:
        Path: Path to each data file found under the sub-directories.
    """
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
    """
    Validate that a data file has the expected extension for its dataset.

    Args:
        file (Path): Path to the file to validate. The parent directory name
            is used to look up the expected extension.

    Raises:
        ValueError: If the file extension does not match the expected
            extension for the dataset type.
    """
    dataset_name = file.parent.stem
    expected_extension = PRE_TRAINING_DATASETS_EXTENSIONS[dataset_name]
    if file.suffix != expected_extension:
        raise ValueError(f"Pretraining file's extension under directory '{dataset_name}' do not match the\
                        correct extension mapping: {expected_extension}")


def load_pre_training_raw_texts(file: Path) -> list[str]:
    """
    Load raw texts from a pre-training data file using the mapped loader.

    Dispatches to the appropriate loading function based on the parent
    directory name of the file.

    Args:
        file (Path): Path to a validated pre-training data file.

    Returns:
        list[str]: Raw text strings extracted from the file.
    """
    dataset_name = file.parent.stem
    return PRE_TRAINING_DATASETS_FN[dataset_name](
        file,
        *PRE_TRAINING_DATASETS_ARGS[dataset_name]
    )


def count_pre_training_tokens(data_path: Union[str, Path],
                              train_split_size: float,
                              tokenizer: Tokenizer,
                              random_seed: int | None = None) -> SplitTokenCounts:
    """
    Count total train and validation tokens across all pre-training files.

    Tokenizes every text in every file under data_path, applies the
    train/validation split, and sums the resulting token counts (including
    one end-of-sequence token per text).

    Args:
        data_path (str | Path): Root data directory (passed to
            get_pre_training_dataset_files).
        train_split_size (float): Fraction of texts to assign to the train
            split; must be in (0, 1].
        tokenizer (Tokenizer): Tokenizer used to encode each text.
        random_seed (int | None): Seed for the internal RNG that controls
            file and text ordering. Pass None for a non-deterministic order.

    Returns:
        SplitTokenCounts: Dataclass with train_tokens and val_tokens fields.
    """
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
    """
    Streaming pre-training dataset with train/validation splitting.

    Reads data files lazily, tokenizes them, and maintains separate token
    buffers for the train and validation splits. Supports both overlapping
    (get_batch) and non-overlapping sequential (get_sequential_batch) access
    patterns.
    """

    def __init__(self,
                 data_path: str,
                 sequence_length: int,
                 train_split_size: float,
                 batch_size: int,
                 tokenizer: Tokenizer,
                 device: torch.device,
                 random_seed: int | None = None) -> None:
        """
        Initialise the pre-training dataset.

        Args:
            data_path (str): Root directory containing dataset sub-directories.
            sequence_length (int): Length of each token sequence in a batch.
            train_split_size (float): Fraction of texts used for training;
                the remainder forms the validation split.
            batch_size (int): Number of sequences per batch.
            tokenizer (Tokenizer): Tokenizer used to encode raw text.
            device (torch.device): Device on which batch tensors are created.
            random_seed (int | None): Optional RNG seed for reproducibility.
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
        Yield tokenized train and validation corpora file by file.

        For each file under data_path, loads and tokenizes all texts, applies
        the train/validation split, concatenates the token sequences (appending
        an EOS token after each text), and yields the two flat token lists.

        Yields:
            tuple[list[int], list[int]]: A ``(train_tokens, val_tokens)`` pair
                of flat token ID lists for one data file.
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
        Check whether the token buffer for a split holds enough tokens for a batch.

        Args:
            buffer_type (str): Either ``"train"`` or ``"val"``.
            required_tokens (int | None): Minimum token count required. Defaults
                to batch_size + sequence_length when None.

        Returns:
            bool: True if the buffer contains at least required_tokens tokens.
        """
        assert buffer_type in ["train", "val"], f"Buffer type must be equal to 'train' or 'val'"
        if required_tokens is None:
            required_tokens = self.batch_size + self.sequence_length
        return len(self.tokens_buffer[buffer_type]) >= required_tokens
    
    def __fill_tokens_buffer(self,
                             split: str,
                             required_tokens: int | None = None) -> None:
        """
        Fill a split's token buffer until it is large enough for a batch.

        Repeatedly draws from the tokenized-splits generator until the buffer
        reaches the required size or the data is exhausted. Sets
        training_data_finished to True when the generator is empty.

        Args:
            split (str): Either ``"train"`` or ``"val"``.
            required_tokens (int | None): Token count target. Forwarded to
                __check_buffer_size.
        """
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
        Sample a batch of overlapping sequences from the token buffer.

        Each call advances the buffer by one token per sequence (sliding-window
        style), so tokens are reused across calls.

        Args:
            split (str): Either ``"train"`` or ``"val"``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(inputs, targets)`` tensors of
                shape (batch_size, sequence_length), where targets is inputs
                shifted right by one position.

        Raises:
            StopIteration: When the data is exhausted and the buffer is too
                small to form a complete batch.
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
        Sample a non-overlapping batch from the token buffer.

        Each call advances the buffer by sequence_length tokens per sequence,
        so every token is used as an input at most once per dataset pass.

        Args:
            split (str): Either ``"train"`` or ``"val"``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(inputs, targets)`` tensors of
                shape (batch_size, sequence_length), where targets is inputs
                shifted right by one position.

        Raises:
            StopIteration: When the data is exhausted and the buffer is too
                small to form a complete batch.
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


class TokenizedDataset:
    """
    Pre-tokenized dataset backed by numpy memmap binary files.

    Reads train.bin and val.bin produced by src.utils.tokenize_dataset.
    Each file is a flat stream of uint32 token IDs. Batches are drawn
    non-overlapping with a persistent offset cursor per split; call
    reset_split() to rewind to the beginning.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        sequence_length: int,
        batch_size: int,
    ) -> None:
        data_dir = Path(data_dir)
        meta_path = data_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"metadata.json not found in {data_dir}. "
                "Run `python -m src.utils.tokenize_dataset` first."
            )

        self.sequence_length = sequence_length
        self.batch_size = batch_size

        self._mmap: dict[str, np.ndarray] = {
            "train": np.memmap(data_dir / "train.bin", dtype=np.uint32, mode="r"),
            "val":   np.memmap(data_dir / "val.bin",   dtype=np.uint32, mode="r"),
        }
        self._offset: dict[str, int] = {"train": 0, "val": 0}

    def reset_split(self, split: str) -> None:
        """Rewind the read cursor for the given split to the beginning."""
        self._offset[split] = 0

    def get_sequential_batch(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return the next non-overlapping batch from the given split.

        Raises:
            StopIteration: When fewer than batch_size * sequence_length + 1
                tokens remain.
        """
        assert split in ("train", "val")
        tokens_needed = self.batch_size * self.sequence_length + 1
        start = self._offset[split]
        mmap = self._mmap[split]

        if start + tokens_needed > len(mmap):
            raise StopIteration

        # np.array() copies the slice out of the memmap so we don't keep the
        # page pinned; cast to int64 for torch embedding look-ups.
        chunk = np.array(mmap[start : start + tokens_needed], dtype=np.int64)

        x = torch.from_numpy(chunk[:-1].reshape(self.batch_size, self.sequence_length))
        y = torch.from_numpy(chunk[1:].reshape(self.batch_size, self.sequence_length))

        self._offset[split] += self.batch_size * self.sequence_length
        return x, y


class FinetuneDataset:
    """
    Finetuning dataset backed by variable-length tokenized session files.

    Reads {split}.bin and {split}_offsets.npy produced by
    src.utils.tokenize_finetune.  Each sample is one tokenized conversation
    chunk.  Samples shorter than sequence_length are right-padded with
    pad_token_id.  Because the causal mask prevents real tokens from attending
    to later positions, padding at the tail never influences real-token
    representations, so no loss mask is needed.

    Shuffling is done at the sample level: sample indices are permuted once at
    construction time (and again on each reset() call).  This ensures every
    sample is seen exactly once per epoch in a randomised order, which
    stabilises gradient variance during finetuning far better than sequential
    access.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        sequence_length: int,
        batch_size: int,
        pad_token_id: int,
        split: str = "train",
        seed: int = 42,
    ) -> None:
        """
        Args:
            data_dir: Directory containing the binary files from tokenize_finetune.
            sequence_length: Model context length; samples are padded/truncated
                to this length.
            batch_size: Number of samples per batch.
            pad_token_id: Token ID used to fill positions beyond the real sample
                length.  The loss mask zeroes out these positions so padding
                never contributes to the gradient.
            split: ``"train"`` or ``"val"``.
            seed: RNG seed for the initial sample-level shuffle.
        """
        data_dir = Path(data_dir)
        meta_path = data_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"metadata.json not found in {data_dir}. "
                "Run `python -m src.utils.tokenize_finetune` first."
            )

        assert split in ("train", "val"), "split must be 'train' or 'val'"

        self.sequence_length = sequence_length
        self.batch_size      = batch_size
        self.pad_token_id    = pad_token_id

        self._tokens:  np.ndarray = np.memmap(
            data_dir / f"{split}.bin", dtype=np.uint32, mode="r"
        )
        self._offsets: np.ndarray = np.load(data_dir / f"{split}_offsets.npy")
        self._n_samples = len(self._offsets) - 1

        # Sample-level shuffle: permute indices so every epoch sees a different
        # order while still visiting every sample exactly once.
        self._order:  np.ndarray = np.random.default_rng(seed).permutation(self._n_samples)
        self._cursor: int = 0

    @property
    def n_samples(self) -> int:
        """Total number of samples in this split."""
        return self._n_samples

    def reset(self, seed: int | None = None) -> None:
        """
        Rewind to the start of the dataset and optionally reshuffle.

        Call between epochs to get a fresh random ordering:
            dataset.reset(seed=epoch_number)
        """
        if seed is not None:
            self._order = np.random.default_rng(seed).permutation(self._n_samples)
        self._cursor = 0

    def get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return the next shuffled batch of ``(inputs, targets)``.

        ``inputs``  — (batch_size, sequence_length) int64
        ``targets`` — (batch_size, sequence_length) int64, shifted right by 1

        Padding tokens sit at the end of each sequence; because the causal mask
        prevents real tokens from attending to later positions, they never see
        the padding and no attention pollution occurs.  Loss is computed
        uniformly over all positions including padding.

        Raises:
            StopIteration: When fewer than batch_size samples remain in the
                current epoch.
        """
        if self._cursor + self.batch_size > self._n_samples:
            raise StopIteration

        xs, ys = [], []
        for i in range(self.batch_size):
            sample_idx = int(self._order[self._cursor + i])
            x, y = self._build_sample(sample_idx)
            xs.append(x)
            ys.append(y)

        self._cursor += self.batch_size
        return torch.stack(xs), torch.stack(ys)

    def _build_sample(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load sample *idx*, pad it to ``sequence_length + 1``, and return
        aligned ``(input, target)`` tensors of length ``sequence_length``.
        """
        start = int(self._offsets[idx])
        end   = int(self._offsets[idx + 1])
        tokens = np.array(self._tokens[start:end], dtype=np.int64)

        needed = self.sequence_length + 1
        padded = np.full(needed, self.pad_token_id, dtype=np.int64)
        padded[:min(len(tokens), needed)] = tokens[:needed]

        return (
            torch.from_numpy(padded[:-1]),  # inputs  (sequence_length,)
            torch.from_numpy(padded[1:]),   # targets (sequence_length,)
        )

