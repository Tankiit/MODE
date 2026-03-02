"""
Simple training data loader for single GPU training.

This is a simplified version of the data loader from train.py that provides
easy access to training batches without the complexity of the full training loop.

Usage:
    from data.simple_train_loader import SimpleTrainLoader

    # Create loader
    loader = SimpleTrainLoader(
        filename_pattern="data/fineweb10B_*.bin",
        num_tokens=512,  # tokens per batch
        max_seq_len=512,
        align_to_bos=True,
    )

    # Get batches in training loop
    for inputs, targets in loader:
        # inputs: (num_tokens,) int32 tensor
        # targets: (num_tokens,) int64 tensor
        loss = model(inputs, targets)
        ...
"""

from pathlib import Path
import glob
from itertools import cycle
import torch

BOS_ID = 50256


def _load_data_shard(file: Path) -> torch.Tensor:
    """Load a single data shard from disk."""
    # Read header and tokens directly to CPU (file I/O requires CPU)
    with file.open("rb", buffering=0) as f:
        # Read header
        header_bytes = f.read(256 * 4)
        header = torch.frombuffer(header_bytes, dtype=torch.int32)

        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        num_tokens = int(header[2])

        # Read tokens
        token_bytes = f.read(2 * num_tokens)
        tokens = torch.frombuffer(token_bytes, dtype=torch.uint16).clone()

    return tokens


class BOSFinder:
    """Helper for getting sequences that start at the beginning of documents."""

    def __init__(self, tokens: torch.Tensor):
        """Precompute BOS positions for the given tokens."""
        self.tokens = tokens
        self.size = tokens.numel()
        self.bos_idx = (
            (tokens == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
        )
        self.i = 0

    def next_batch(self, num_tokens_local: int, max_seq_len: int):
        """
        Get the next batch aligned to BOS tokens.

        Args:
            num_tokens_local: Total number of tokens needed
            max_seq_len: Maximum sequence length

        Returns:
            tuple: (start_indices, end_indices) for the batch
        """
        n = len(self.bos_idx)
        starts = []
        ends = []

        idx = self.i
        cur_len = 0
        while cur_len <= num_tokens_local:
            if idx >= n:
                raise StopIteration("Insufficient BOS ahead; hit tail of shard.")
            cur = self.bos_idx[idx]
            starts.append(cur)
            end = min(
                self.bos_idx[idx + 1] if idx + 1 < n else self.size,
                cur + max_seq_len,
                cur + num_tokens_local - cur_len + 1,
            )
            ends.append(end)
            cur_len += end - cur
            idx += 1

        assert cur_len == num_tokens_local + 1
        self.i = idx
        return starts, ends


class SimpleTrainLoader:
    """
    Simple training data loader for single GPU training.

    Handles loading tokens from .bin files and yielding batches for training.
    Supports both BOS-aligned batching (for document boundaries) and simple
    sequential batching.
    """

    def __init__(
        self,
        filename_pattern: str,
        num_tokens: int,
        max_seq_len: int,
        align_to_bos: bool = True,
        device: torch.device | None = None,
    ):
        """
        Initialize the data loader.

        Args:
            filename_pattern: Glob pattern for data files (e.g., "data/*.bin")
            num_tokens: Number of tokens per batch
            max_seq_len: Maximum sequence length
            align_to_bos: If True, align batches to document boundaries (BOS tokens)
            device: Device to load tensors to (default: auto-detect)
        """
        if device is None:
            device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else (
                    "mps"
                    if getattr(torch.backends, "mps", None) is not None
                    and torch.backends.mps.is_available()
                    else "cpu"
                )
            )

        self.device = device
        self.num_tokens = num_tokens
        self.max_seq_len = max_seq_len
        self.align_to_bos = align_to_bos

        # Find all matching files
        files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
        if not files:
            raise FileNotFoundError(f"No data shards matched pattern: {filename_pattern}")

        self.files = files
        self.file_iter = cycle(files)

        # Load first shard
        self.tokens = _load_data_shard(next(self.file_iter))

        if align_to_bos:
            self.finder = BOSFinder(self.tokens)
        else:
            self.pos = 0

    def __iter__(self):
        """Return iterator interface."""
        return self

    def __next__(self):
        """Get the next batch."""
        if self.align_to_bos:
            return self._next_bos_aligned()
        else:
            return self._next_sequential()

    def _next_bos_aligned(self):
        """Get next batch aligned to BOS tokens (document boundaries)."""
        try:
            start_idxs, end_idxs = self.finder.next_batch(
                self.num_tokens, self.max_seq_len
            )
        except StopIteration:
            # Move to next shard
            self.tokens = _load_data_shard(next(self.file_iter))
            self.finder = BOSFinder(self.tokens)
            start_idxs, end_idxs = self.finder.next_batch(
                self.num_tokens, self.max_seq_len
            )

        # Concatenate sequences
        buf = torch.cat([self.tokens[i:j] for i, j in zip(start_idxs, end_idxs)])

        # Split into inputs and targets
        inputs = buf[:-1].to(device=self.device, dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device=self.device, dtype=torch.int64, non_blocking=True)

        return inputs, targets

    def _next_sequential(self):
        """Get next batch sequentially (no BOS alignment)."""
        if self.pos + self.num_tokens + 1 >= len(self.tokens):
            # Move to next shard
            self.tokens = _load_data_shard(next(self.file_iter))
            self.pos = 0

        buf = self.tokens[self.pos : self.pos + self.num_tokens + 1]
        self.pos += self.num_tokens

        inputs = buf[:-1].to(device=self.device, dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device=self.device, dtype=torch.int64, non_blocking=True)

        return inputs, targets


def create_simple_loader(
    filename_pattern: str,
    batch_size_tokens: int,
    seq_len: int,
    device: torch.device | None = None,
) -> SimpleTrainLoader:
    """
    Convenience function to create a SimpleTrainLoader.

    Args:
        filename_pattern: Glob pattern for data files
        batch_size_tokens: Number of tokens per batch
        seq_len: Maximum sequence length
        device: Device to load tensors to

    Returns:
        SimpleTrainLoader instance

    Example:
        >>> loader = create_simple_loader("data/*.bin", batch_size_tokens=512, seq_len=512)
        >>> inputs, targets = next(loader)
    """
    return SimpleTrainLoader(
        filename_pattern=filename_pattern,
        num_tokens=batch_size_tokens,
        max_seq_len=seq_len,
        align_to_bos=True,
        device=device,
    )
