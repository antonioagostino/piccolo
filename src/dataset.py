from pathlib import Path
from typing import Union
import numpy as np
import torch

class TokenizedPreTrainingDataset:
    """Pre-tokenized dataset for pre-training, backed by numpy memmap binary files."""
    def __init__(
        self,
        data_dir: Union[str, Path],
        sequence_length: int,
        batch_size: int,
    ) -> None:
        data_dir = Path(data_dir)
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

    def reset_epoch(self, _seed: int | None = None) -> None:
        """Rewind the training cursor to start a new epoch.

        The _seed parameter is accepted for API compatibility with
        FinetuneDataset but is ignored — TokenizedDataset reads tokens in a
        fixed sequential order and has nothing to reshuffle.
        """
        self.reset_split("train")

    def get_sequential_batch(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the next non-overlapping batch from the given split."""
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


class TokenizedFinetuneDataset:
    """Finetuning dataset backed by variable-length tokenized sample files."""

    def __init__(
        self,
        data_dir: Union[str, Path],
        sequence_length: int,
        batch_size: int,
        pad_token_id: int,
        seed: int = 42,
    ) -> None:
        data_dir = Path(data_dir)

        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.pad_token_id = pad_token_id

        self._tokens: dict[str, np.ndarray] = {
            "train": np.memmap(
                data_dir / "train.bin", dtype=np.uint32, mode="r"
            ),
            "val": np.memmap(
                data_dir / "val.bin", dtype=np.uint32, mode="r"
            )
        }
        self._offsets: dict[str, np.ndarray] = {
            "train": np.load(data_dir / "train_offsets.npy"),
            "val": np.load(data_dir / "val_offsets.npy")
        }
        self.n_samples: dict[str, int] = {
            "train": len(self._offsets["train"]) - 1,
            "val": len(self._offsets["val"]) - 1,
        }

        # Sample-level shuffle: permute indices so every epoch sees a different
        # order while still visiting every sample exactly once.
        self._orders: dict[str, np.ndarray] = {
            "train": np.random.default_rng(seed).permutation(self.n_samples["train"]),
            "val": np.random.default_rng(seed).permutation(self.n_samples["val"]),
        }
        self._cursors: dict[str, int] = {
            "train": 0,
            "val": 0
        }

    def reset(self, split: str, seed: int | None = None) -> None:
        """Rewind to the start of the dataset and optionally reshuffle."""
        assert split in ["train", "val"], "The split must be 'train' or 'val'"
        if seed is not None:
            self._orders[split] = np.random.default_rng(seed).permutation(self.n_samples[split])
        self._cursors[split] = 0

    def reset_epoch(self, split: str, seed: int | None = None) -> None:
        """Reshuffle and rewind for a new epoch."""
        assert split in ["train", "val"], "The split must be 'train' or 'val'"
        self.reset(split=split, seed=seed)

    def reset_split(self, split: str) -> None:
        """Shim for TokenizedDataset API compatibility (used by validate())."""
        assert split in ["train", "val"], "The split must be 'train' or 'val'"
        self._cursors[split] = 0

    def get_sequential_batch(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Shim for TokenizedDataset API compatibility (used by validate())."""
        assert split in ["train", "val"], "The split must be 'train' or 'val'"
        return self.get_batch(split)

    def get_batch(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the next shuffled batch of ``(inputs, targets)``."""
        assert split in ["train", "val"], "The split must be 'train' or 'val'"
        if self._cursors[split] + self.batch_size > self.n_samples[split]:
            raise StopIteration

        xs, ys = [], []
        for i in range(self.batch_size):
            sample_idx = int(self._orders[split][self._cursors[split] + i])
            start = int(self._offsets[split][sample_idx])
            end = int(self._offsets[split][sample_idx + 1])
            tokens = np.array(self._tokens[split][start:end], dtype=np.int64)

            needed = self.sequence_length + 1
            padded = np.full(needed, self.pad_token_id, dtype=np.int64)
            padded[:min(len(tokens), needed)] = tokens[:needed]

            x = torch.from_numpy(padded[:-1])  # inputs  (sequence_length,)
            y = torch.from_numpy(padded[1:])   # targets (sequence_length,)

            xs.append(x)
            ys.append(y)

        self._cursors[split] += self.batch_size
        return torch.stack(xs), torch.stack(ys)
