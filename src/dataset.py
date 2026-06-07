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


class FinetuneDataset:
    """
    Finetuning dataset backed by variable-length tokenized sample files.

    Expects the following files in data_dir:
        {split}.bin          flat uint32 array of token ids, all samples concatenated
        {split}_offsets.npy  int64 array of shape (n_samples + 1,) marking
                             where each sample starts in the .bin file
        metadata.json        optional, used for validation at construction time

    Samples shorter than sequence_length are right-padded with pad_token_id.
    Because the causal mask prevents real tokens from attending to later
    positions, padding at the tail never influences real-token representations,
    so no loss mask is needed.

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
            data_dir: Directory containing {split}.bin and {split}_offsets.npy.
            sequence_length: Model context length; samples are padded/truncated
                to this length.
            batch_size: Number of samples per batch.
            pad_token_id: Token ID used to fill positions beyond the real sample
                length.
            split: ``"train"`` or ``"val"``.
            seed: RNG seed for the initial sample-level shuffle.
        """
        data_dir = Path(data_dir)
        meta_path = data_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"metadata.json not found in {data_dir}. "
                "Make sure the tokenization script has been run first."
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

    def reset_epoch(self, seed: int | None = None) -> None:
        """
        Reshuffle and rewind for a new epoch.

        Pass a seed that varies per epoch to guarantee a different sample
        order each pass:
            dataset.reset_epoch(seed=base_seed + epoch)
        """
        self.reset(seed=seed)

    def reset_split(self, _split: str) -> None:
        """
        Shim for TokenizedDataset API compatibility (used by validate()).

        Rewinds the read cursor without reshuffling so validation is
        deterministic. The split argument is ignored — this FinetuneDataset
        instance is already bound to one split at construction time.
        """
        self._cursor = 0

    def get_sequential_batch(self, _split: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Shim for TokenizedDataset API compatibility (used by validate()).

        Delegates to get_batch(); the split argument is ignored because
        this FinetuneDataset instance is already bound to one split.
        """
        return self.get_batch()

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

