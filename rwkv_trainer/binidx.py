"""Read-only RWKV binidx reader and deterministic stateful TorchTitan loader."""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.tokenizer import BaseTokenizer

_DTYPES = {1: np.uint8, 2: np.int8, 3: np.int16, 4: np.int32, 5: np.int64, 8: np.uint16}
_MAGIC = b"MMIDIDX\x00\x00"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    divisor = 3
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 2
    return True


class BinidxTokens:
    def __init__(self, prefix: str | Path):
        self.prefix = Path(prefix)
        self.index_path = Path(f"{self.prefix}.idx")
        self.data_path = Path(f"{self.prefix}.bin")
        if not self.index_path.is_file() or not self.data_path.is_file():
            raise FileNotFoundError(f"binidx requires {self.index_path} and {self.data_path}")
        with self.index_path.open("rb") as stream:
            if stream.read(9) != _MAGIC:
                raise ValueError(f"invalid binidx header: {self.index_path}")
            if struct.unpack("<Q", stream.read(8))[0] != 1:
                raise ValueError("only binidx index version 1 is supported")
            code = struct.unpack("<B", stream.read(1))[0]
            if code not in _DTYPES:
                raise ValueError(f"unsupported binidx dtype code: {code}")
            self.dtype = _DTYPES[code]
            length = struct.unpack("<Q", stream.read(8))[0]
            documents = struct.unpack("<Q", stream.read(8))[0]
            offset = stream.tell()
        index_map = np.memmap(self.index_path, mode="r")
        self._index_map = index_map
        self.sizes = np.frombuffer(index_map, dtype=np.int32, count=length, offset=offset)
        pointer_offset = offset + self.sizes.nbytes
        self.pointers = np.frombuffer(
            index_map, dtype=np.int64, count=length, offset=pointer_offset
        )
        document_offset = pointer_offset + self.pointers.nbytes
        self.documents = np.frombuffer(
            index_map, dtype=np.int64, count=documents, offset=document_offset
        )
        if length != 1:
            raise ValueError("RWKV v1 binidx requires one contiguous token item.")
        self._data_map = np.memmap(self.data_path, mode="r")
        self.identity = {
            "index_sha256": _sha256(self.index_path),
            "data_sha256": _sha256(self.data_path),
            "tokens": int(self.sizes[0]),
            "dtype": np.dtype(self.dtype).str,
        }

    def get(self, offset: int, length: int) -> np.ndarray:
        size = int(self.sizes[0])
        if offset < 0 or length <= 0 or offset + length > size:
            raise IndexError(f"binidx token range [{offset}, {offset + length}) exceeds {size}")
        byte_offset = int(self.pointers[0]) + offset * np.dtype(self.dtype).itemsize
        return np.frombuffer(self._data_map, dtype=self.dtype, count=length, offset=byte_offset)


class RwkvDataLoader(BaseDataLoader):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        dataset: str = "synthetic"
        vocab_size: int = 1024
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
        snapshot_every_n_steps: int | None = None,
        **kwargs: Any,
    ):
        del tokenizer, snapshot_every_n_steps
        if kwargs:
            raise TypeError(f"unsupported RWKV dataloader options: {sorted(kwargs)}")
        if dp_world_size <= 0 or not 0 <= dp_rank < dp_world_size:
            raise ValueError("invalid data-parallel rank/world size")
        if seq_len <= 0 or local_batch_size <= 0:
            raise ValueError("seq_len and local_batch_size must be positive")
        self.config = config
        self.dp_world_size = dp_world_size
        self.dp_rank = dp_rank
        self.seq_len = seq_len
        self.local_batch_size = local_batch_size
        self.cursor = 0
        if config.dataset not in {"synthetic", "binidx"}:
            raise ValueError(f"unsupported RWKV dataset: {config.dataset}")
        if config.dataset == "binidx" and config.dataset_path is None:
            raise ValueError("binidx dataset requires dataset_path")
        if config.dataset == "binidx":
            assert config.dataset_path is not None
            self.data = BinidxTokens(config.dataset_path)
        else:
            self.data = None
        if self.data is not None:
            slots = (self.data.identity["tokens"] - 1) // seq_len
            prime = config.magic_prime
            if prime is None or not _is_prime(prime) or prime % 3 != 2:
                raise ValueError("binidx magic_prime must be prime and congruent to 2 modulo 3")
            if not 0.9 < prime / slots <= 1:
                raise ValueError(
                    "magic_prime must cover more than 90% and at most 100% of sequence slots"
                )

    def __iter__(self) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        while self.config.infinite or self.cursor < self.config.num_batches:
            tokens = self._tokens()
            positions = torch.arange(self.seq_len, dtype=torch.long).expand(
                self.local_batch_size, -1
            )
            self.cursor += 1
            yield {"input": tokens[:, :-1], "positions": positions}, tokens[:, 1:]

    def _tokens(self) -> torch.Tensor:
        if self.data is None:
            generator = torch.Generator().manual_seed(
                self.config.seed + self.cursor * self.dp_world_size + self.dp_rank
            )
            return torch.randint(
                self.config.vocab_size,
                (self.local_batch_size, self.seq_len + 1),
                generator=generator,
            )
        prime = self.config.magic_prime
        assert prime is not None
        factor = int(prime * ((math.sqrt(5) - 1) / 2))
        rows = []
        for batch_index in range(self.local_batch_size):
            sample = (
                (self.cursor * self.local_batch_size + batch_index) * self.dp_world_size
                + self.dp_rank
                + 1
            )
            slot = (factor * sample**3) % prime
            rows.append(
                torch.from_numpy(
                    np.array(self.data.get(slot * self.seq_len, self.seq_len + 1), dtype=np.int64)
                )
            )
        return torch.stack(rows)

    def _identity(self) -> dict[str, Any]:
        return {
            "dataset": self.config.dataset,
            "data": None if self.data is None else self.data.identity,
            "vocab_size": self.config.vocab_size,
            "seed": self.config.seed,
            "magic_prime": self.config.magic_prime,
            "dp_world_size": self.dp_world_size,
            "seq_len": self.seq_len,
            "local_batch_size": self.local_batch_size,
        }

    def state_dict(self) -> dict[str, Any]:
        return {"cursor": self.cursor, "identity": self._identity()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not state_dict:
            return
        if state_dict.get("identity") != self._identity():
            raise ValueError("RWKV dataloader checkpoint identity mismatch")
        cursor = state_dict.get("cursor")
        if not isinstance(cursor, int) or cursor < 0:
            raise ValueError("RWKV dataloader cursor must be a non-negative integer")
        self.cursor = cursor


__all__ = ["BinidxTokens", "RwkvDataLoader"]
