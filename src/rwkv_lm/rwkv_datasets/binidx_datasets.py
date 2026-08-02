"""Memory-mapped RWKV ``.bin``/``.idx`` token datasets."""

from __future__ import annotations

import struct
from itertools import accumulate
from pathlib import Path
from types import TracebackType
from typing import Self

import numpy as np
from torch.utils.data import Dataset

_DTYPES = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float32,
    7: np.float64,
    8: np.uint16,
}


def index_file_path(prefix_path: str | Path) -> Path:
    """Return the index path for a binidx prefix."""
    return Path(f"{prefix_path}.idx")


def data_file_path(prefix_path: str | Path) -> Path:
    """Return the token-data path for a binidx prefix."""
    return Path(f"{prefix_path}.bin")


def _dtype_code(dtype: type[np.generic]) -> int:
    for code, candidate in _DTYPES.items():
        if candidate is dtype:
            return code
    raise ValueError(f"unsupported binidx dtype: {dtype}")


class _IndexWriter:
    def __init__(self, path: str | Path, dtype: type[np.generic]) -> None:
        self.path = Path(path)
        self.dtype = dtype
        self._file = None

    def __enter__(self) -> Self:
        self._file = self.path.open("wb")
        self._file.write(MMapIndexedDataset.Index.HEADER_MAGIC)
        self._file.write(struct.pack("<Q", 1))
        self._file.write(struct.pack("<B", _dtype_code(self.dtype)))
        return self

    def write(self, sizes: list[int], document_indices: list[int]) -> None:
        if self._file is None:
            raise RuntimeError("binidx index writer is not open")
        item_size = np.dtype(self.dtype).itemsize
        pointers = []
        address = 0
        for size in sizes:
            pointers.append(address)
            address += size * item_size
        self._file.write(struct.pack("<Q", len(sizes)))
        self._file.write(struct.pack("<Q", len(document_indices)))
        self._file.write(np.asarray(sizes, dtype=np.int32).tobytes(order="C"))
        self._file.write(np.asarray(pointers, dtype=np.int64).tobytes(order="C"))
        self._file.write(
            np.asarray(document_indices, dtype=np.int64).tobytes(order="C")
        )

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        if self._file is not None:
            self._file.close()


class MMapIndexedDataset(Dataset):
    """Read-only token dataset compatible with the legacy RWKV binidx format."""

    class Index:
        HEADER_MAGIC = b"MMIDIDX\x00\x00"

        @classmethod
        def writer(
            cls,
            path: str | Path,
            dtype: type[np.generic],
        ) -> _IndexWriter:
            del cls
            return _IndexWriter(path, dtype)

        def __init__(self, path: str | Path) -> None:
            self.path = Path(path)
            with self.path.open("rb") as stream:
                if stream.read(9) != self.HEADER_MAGIC:
                    raise ValueError(f"invalid binidx header: {self.path}")
                version = struct.unpack("<Q", stream.read(8))[0]
                if version != 1:
                    raise ValueError(f"unsupported binidx version: {version}")
                dtype_code = struct.unpack("<B", stream.read(1))[0]
                try:
                    self.dtype = _DTYPES[dtype_code]
                except KeyError as error:
                    raise ValueError(
                        f"unsupported binidx dtype code: {dtype_code}"
                    ) from error
                self.length = struct.unpack("<Q", stream.read(8))[0]
                document_count = struct.unpack("<Q", stream.read(8))[0]
                offset = stream.tell()

            self._mmap = np.memmap(self.path, mode="r", order="C")
            self._buffer = memoryview(self._mmap)
            self.sizes = np.frombuffer(
                self._buffer,
                dtype=np.int32,
                count=self.length,
                offset=offset,
            )
            self.pointers = np.frombuffer(
                self._buffer,
                dtype=np.int64,
                count=self.length,
                offset=offset + self.sizes.nbytes,
            )
            self.document_indices = np.frombuffer(
                self._buffer,
                dtype=np.int64,
                count=document_count,
                offset=offset + self.sizes.nbytes + self.pointers.nbytes,
            )

        def close(self) -> None:
            self._buffer.release()
            self._mmap._mmap.close()

        def __getitem__(self, index: int) -> tuple[int, int]:
            return int(self.pointers[index]), int(self.sizes[index])

        def __len__(self) -> int:
            return self.length

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self.path = Path(path)
        if not self.exists(self.path):
            raise FileNotFoundError(
                f"binidx dataset requires {data_file_path(self.path)} and "
                f"{index_file_path(self.path)}"
            )
        self.index = self.Index(index_file_path(self.path))
        self._mmap = np.memmap(data_file_path(self.path), mode="r", order="C")
        self._buffer = memoryview(self._mmap)

    def close(self) -> None:
        self._buffer.release()
        self._mmap._mmap.close()
        self.index.close()

    def __getstate__(self) -> Path:
        return self.path

    def __setstate__(self, path: Path) -> None:
        self.__init__(path)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int | slice) -> np.ndarray | list[np.ndarray]:
        if isinstance(index, int):
            pointer, size = self.index[index]
            return np.frombuffer(
                self._buffer,
                dtype=self.index.dtype,
                count=size,
                offset=pointer,
            )
        start, stop, step = index.indices(len(self))
        if step != 1:
            raise ValueError("binidx slices must be contiguous")
        if start == stop:
            return []
        pointer = int(self.index.pointers[start])
        sizes = self.index.sizes[index]
        offsets = list(accumulate(int(size) for size in sizes))
        values = np.frombuffer(
            self._buffer,
            dtype=self.index.dtype,
            count=sum(int(size) for size in sizes),
            offset=pointer,
        )
        return list(np.split(values, offsets[:-1]))

    def get(
        self, index: int, *, offset: int = 0, length: int | None = None
    ) -> np.ndarray:
        """Read a contiguous token range from one indexed item."""
        pointer, size = self.index[index]
        if offset < 0 or offset > size:
            raise IndexError(f"binidx offset {offset} is outside item of size {size}")
        if length is None:
            length = size - offset
        if length < 0 or offset + length > size:
            raise IndexError(
                f"binidx range [{offset}, {offset + length}) exceeds item size {size}"
            )
        pointer += offset * np.dtype(self.index.dtype).itemsize
        return np.frombuffer(
            self._buffer,
            dtype=self.index.dtype,
            count=length,
            offset=pointer,
        )

    @property
    def item_sizes(self) -> np.ndarray:
        return self.index.sizes

    @property
    def document_indices(self) -> np.ndarray:
        return self.index.document_indices

    @staticmethod
    def exists(path: str | Path) -> bool:
        return index_file_path(path).is_file() and data_file_path(path).is_file()


__all__ = ["MMapIndexedDataset", "data_file_path", "index_file_path"]
