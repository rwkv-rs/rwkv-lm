"""TorchTitan dataloader for RWKV synthetic and binidx text data."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.tokenizer import BaseTokenizer

from .binidx_datasets import MMapIndexedDataset


def _is_prime(value: int) -> bool:
    if value <= 1:
        return False
    if value <= 3:
        return True
    if value % 2 == 0 or value % 3 == 0:
        return False
    divisor = 5
    while divisor * divisor <= value:
        if value % divisor == 0 or value % (divisor + 2) == 0:
            return False
        divisor += 6
    return True


class RwkvDataLoader(BaseDataLoader):
    """Stateful synthetic or legacy-binidx batches owned by TorchTitan Trainer."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        dataset: Literal["synthetic", "binidx"] = "synthetic"
        vocab_size: int = 1_024
        seed: int = 42
        magic_prime: int | None = None
        infinite: bool = True
        num_batches: int = 10

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: BaseTokenizer,
        seq_len: int,
        local_batch_size: int,
        snapshot_every_n_steps: int | None = 1,
        **kwargs: Any,
    ) -> None:
        del tokenizer, snapshot_every_n_steps
        if kwargs:
            raise TypeError(f"unsupported RWKV dataloader options: {sorted(kwargs)}")
        if dp_world_size <= 0 or not 0 <= dp_rank < dp_world_size:
            raise ValueError("invalid RWKV dataloader DP rank/world size")
        if seq_len <= 0 or local_batch_size <= 0:
            raise ValueError("RWKV seq_len and local_batch_size must be positive")
        if config.vocab_size <= 1:
            raise ValueError("RWKV dataloader vocab_size must be greater than one")
        if config.num_batches <= 0:
            raise ValueError("RWKV dataloader num_batches must be positive")

        self.config = config
        self.dp_world_size = dp_world_size
        self.dp_rank = dp_rank
        self.seq_len = seq_len
        self.local_batch_size = local_batch_size
        self._cursor = 0
        self._data: MMapIndexedDataset | None = None
        self._dataset_slots = 0

        if config.dataset == "binidx":
            if not config.dataset_path:
                raise ValueError(
                    "RWKV binidx training requires --dataloader.dataset-path"
                )
            self._data = MMapIndexedDataset(Path(config.dataset_path))
            if len(self._data) != 1:
                raise ValueError(
                    "RWKV binidx training currently requires one contiguous token item"
                )
            token_count = int(self._data.item_sizes[0])
            self._dataset_slots = (token_count - 1) // seq_len
            if self._dataset_slots <= 0:
                raise ValueError(
                    f"RWKV binidx item needs at least {seq_len + 1} tokens"
                )
            if config.magic_prime is None:
                raise ValueError(
                    "RWKV binidx training requires --dataloader.magic-prime"
                )
            if not _is_prime(config.magic_prime) or config.magic_prime % 3 != 2:
                raise ValueError(
                    "RWKV magic_prime must be prime and congruent to 2 mod 3"
                )
            coverage = config.magic_prime / self._dataset_slots
            if not 0.9 < coverage <= 1:
                raise ValueError(
                    "RWKV magic_prime must cover more than 90% and at most 100% "
                    "of binidx sequence slots"
                )

    def __iter__(self) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        while self.config.infinite or self._cursor < self.config.num_batches:
            if self.config.dataset == "synthetic":
                tokens = self._synthetic_tokens()
            else:
                tokens = self._binidx_tokens()
            positions = torch.arange(self.seq_len, dtype=torch.long).expand(
                self.local_batch_size,
                -1,
            )
            self._cursor += 1
            yield {"input": tokens[:, :-1], "positions": positions}, tokens[:, 1:]

    def _synthetic_tokens(self) -> torch.Tensor:
        generator = torch.Generator().manual_seed(
            self.config.seed + self._cursor * self.dp_world_size + self.dp_rank
        )
        return torch.randint(
            0,
            self.config.vocab_size,
            (self.local_batch_size, self.seq_len + 1),
            generator=generator,
            dtype=torch.long,
        )

    def _binidx_tokens(self) -> torch.Tensor:
        assert self._data is not None
        assert self.config.magic_prime is not None
        golden_ratio = (math.sqrt(5) - 1) / 2
        factor = int(self.config.magic_prime * golden_ratio)
        samples = []
        for batch_index in range(self.local_batch_size):
            sample_index = (
                (self._cursor * self.local_batch_size + batch_index)
                * self.dp_world_size
                + self.dp_rank
                + 1
            )
            slot = (
                factor * sample_index * sample_index * sample_index
            ) % self.config.magic_prime
            values = self._data.get(
                0,
                offset=slot * self.seq_len,
                length=self.seq_len + 1,
            )
            samples.append(torch.from_numpy(np.array(values, dtype=np.int64)))
        return torch.stack(samples)

    def state_dict(self) -> dict[str, Any]:
        return {"cursor": self._cursor, **self._checkpoint_identity()}

    def _checkpoint_identity(self) -> dict[str, Any]:
        return {
            "dataset": self.config.dataset,
            "dataset_path": self.config.dataset_path,
            "vocab_size": self.config.vocab_size,
            "seed": self.config.seed,
            "magic_prime": self.config.magic_prime,
            "infinite": self.config.infinite,
            "num_batches": self.config.num_batches,
            "dp_world_size": self.dp_world_size,
            "dp_rank": self.dp_rank,
            "seq_len": self.seq_len,
            "local_batch_size": self.local_batch_size,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not state_dict:
            return
        expected = self._checkpoint_identity()
        observed = {key: state_dict.get(key) for key in expected}
        if observed != expected:
            raise ValueError(
                f"RWKV dataloader checkpoint identity mismatch: "
                f"expected {expected}, got {observed}"
            )
        cursor = state_dict.get("cursor")
        if not isinstance(cursor, int) or cursor < 0:
            raise ValueError("RWKV dataloader checkpoint cursor must be non-negative")
        self._cursor = cursor


__all__ = ["RwkvDataLoader"]
