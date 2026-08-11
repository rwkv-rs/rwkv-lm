"""Deterministic fixture generation and streaming DCP comparison for GPU acceptance."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import BytesStorageMetadata, TensorStorageMetadata
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner

_MAGIC = b"MMIDIDX\x00\x00"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_hashes(path: Path) -> dict[str, str]:
    return {
        str(item.relative_to(path)): _sha256(item)
        for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    }


def make_binidx(args: argparse.Namespace) -> None:
    if args.seq_len != 4096 or args.slots != 1024 or args.magic_prime != 1019:
        raise ValueError(
            "The canonical acceptance fixture fixes seq_len=4096, slots=1024, and magic_prime=1019."
        )
    if not 2 <= args.vocab_size <= 65536:
        raise ValueError("uint16 acceptance tokens require vocab_size in [2, 65536].")
    if args.seed != 20260808:
        raise ValueError("The canonical acceptance fixture fixes seed=20260808.")
    prefix = Path(args.output_prefix)
    index_path = Path(f"{prefix}.idx")
    data_path = Path(f"{prefix}.bin")
    if index_path.exists() or data_path.exists():
        raise FileExistsError(f"Refusing to overwrite {index_path} or {data_path}.")
    prefix.parent.mkdir(parents=True, exist_ok=True)

    token_count = args.seq_len * args.slots + 1
    indices = np.arange(token_count, dtype=np.uint64)
    tokens = ((indices * 48271 + args.seed) % args.vocab_size).astype(np.uint16)
    data_path.write_bytes(tokens.tobytes())
    index = bytearray(_MAGIC)
    index += struct.pack("<Q", 1)
    index += struct.pack("<B", 8)
    index += struct.pack("<Q", 1)
    index += struct.pack("<Q", 1)
    index += np.asarray([token_count], dtype=np.int32).tobytes()
    index += np.asarray([0], dtype=np.int64).tobytes()
    index += np.asarray([0], dtype=np.int64).tobytes()
    index_path.write_bytes(index)

    report = {
        "output_prefix": str(prefix.resolve()),
        "seq_len": args.seq_len,
        "slots": args.slots,
        "tokens": token_count,
        "vocab_size": args.vocab_size,
        "seed": args.seed,
        "magic_prime": args.magic_prime,
        "magic_prime_ratio": args.magic_prime / args.slots,
        "dtype": np.dtype(np.uint16).str,
        "index_sha256": _sha256(index_path),
        "data_sha256": _sha256(data_path),
    }
    print(json.dumps(report, indent=2))


def _allocate(metadata: TensorStorageMetadata | BytesStorageMetadata) -> Any:
    if isinstance(metadata, TensorStorageMetadata):
        return torch.empty(metadata.size, dtype=metadata.properties.dtype, device="cpu")
    return None


def _load_key(
    reader: dcp.FileSystemReader,
    key: str,
    metadata: TensorStorageMetadata | BytesStorageMetadata,
) -> Any:
    state = {key: _allocate(metadata)}
    planner = DefaultLoadPlanner(flatten_state_dict=False, allow_partial_load=True)
    dcp.load(state, storage_reader=reader, planner=planner)
    return state[key]


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()


def _compare_values(
    left: Any,
    right: Any,
    *,
    path: str,
    atol: float,
    rtol: float,
    differences: list[dict[str, Any]],
) -> None:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            differences.append({"path": path, "reason": "tensor type mismatch"})
            return
        if left.shape != right.shape or left.dtype != right.dtype:
            differences.append(
                {
                    "path": path,
                    "reason": "tensor metadata mismatch",
                    "left_shape": list(left.shape),
                    "right_shape": list(right.shape),
                    "left_dtype": str(left.dtype),
                    "right_dtype": str(right.dtype),
                }
            )
            return
        if left.dtype in (torch.bfloat16, torch.float16):
            delta = (left.float() - right.float()).abs()
            allowed = atol + rtol * right.float().abs()
            if not torch.all(delta <= allowed):
                relative = delta / right.float().abs().clamp_min(torch.finfo(torch.float32).tiny)
                differences.append(
                    {
                        "path": path,
                        "reason": "low-precision tensor mismatch",
                        "max_absolute_difference": delta.max().item(),
                        "max_relative_difference": relative.max().item(),
                    }
                )
        elif not torch.equal(left, right):
            differences.append(
                {
                    "path": path,
                    "reason": "bitwise tensor mismatch",
                    "left_sha256": _tensor_sha256(left),
                    "right_sha256": _tensor_sha256(right),
                }
            )
        return
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
            differences.append({"path": path, "reason": "array type mismatch"})
        elif (
            left.dtype != right.dtype
            or left.shape != right.shape
            or not np.array_equal(left, right)
        ):
            differences.append({"path": path, "reason": "numpy array mismatch"})
        return
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            differences.append({"path": path, "reason": "mapping type mismatch"})
            return
        left_keys = set(left)
        right_keys = set(right)
        if left_keys != right_keys:
            differences.append(
                {
                    "path": path,
                    "reason": "mapping key mismatch",
                    "only_left": sorted(map(str, left_keys - right_keys)),
                    "only_right": sorted(map(str, right_keys - left_keys)),
                }
            )
            return
        for key in sorted(left_keys, key=str):
            _compare_values(
                left[key],
                right[key],
                path=f"{path}.{key}",
                atol=atol,
                rtol=rtol,
                differences=differences,
            )
        return
    sequence_types = (list, tuple)
    if isinstance(left, sequence_types) or isinstance(right, sequence_types):
        if not isinstance(left, sequence_types) or not isinstance(right, sequence_types):
            differences.append({"path": path, "reason": "sequence type mismatch"})
            return
        if len(left) != len(right):
            differences.append({"path": path, "reason": "sequence length mismatch"})
            return
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            _compare_values(
                left_item,
                right_item,
                path=f"{path}[{index}]",
                atol=atol,
                rtol=rtol,
                differences=differences,
            )
        return
    if type(left) is not type(right) or left != right:
        differences.append(
            {
                "path": path,
                "reason": "value mismatch",
                "left": repr(left),
                "right": repr(right),
            }
        )


def compare_dcp(args: argparse.Namespace) -> None:
    left_path = Path(args.left)
    right_path = Path(args.right)
    left_reader = dcp.FileSystemReader(left_path)
    right_reader = dcp.FileSystemReader(right_path)
    left_metadata = left_reader.read_metadata().state_dict_metadata
    right_metadata = right_reader.read_metadata().state_dict_metadata
    left_keys = set(left_metadata)
    right_keys = set(right_metadata)
    differences: list[dict[str, Any]] = []
    if left_keys != right_keys:
        differences.append(
            {
                "path": "state_dict",
                "reason": "checkpoint key mismatch",
                "only_left": sorted(left_keys - right_keys),
                "only_right": sorted(right_keys - left_keys),
            }
        )

    compared = 0
    for key in sorted(left_keys & right_keys):
        left_value = _load_key(left_reader, key, left_metadata[key])
        right_value = _load_key(right_reader, key, right_metadata[key])
        _compare_values(
            left_value,
            right_value,
            path=key,
            atol=args.atol,
            rtol=args.rtol,
            differences=differences,
        )
        compared += 1

    report = {
        "left": str(left_path.resolve()),
        "right": str(right_path.resolve()),
        "atol": args.atol,
        "rtol": args.rtol,
        "keys_compared": compared,
        "differences": differences,
        "checkpoint_sha256": {
            "left": _artifact_hashes(left_path),
            "right": _artifact_hashes(right_path),
        },
    }
    output = json.dumps(report, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(output)
    print(output, end="")
    if differences:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    fixture = subparsers.add_parser("make-binidx")
    fixture.add_argument("--output-prefix", required=True)
    fixture.add_argument("--vocab-size", type=int, required=True)
    fixture.add_argument("--seq-len", type=int, default=4096)
    fixture.add_argument("--slots", type=int, default=1024)
    fixture.add_argument("--magic-prime", type=int, default=1019)
    fixture.add_argument("--seed", type=int, default=20260808)
    fixture.set_defaults(function=make_binidx)

    compare = subparsers.add_parser("compare-dcp")
    compare.add_argument("--left", required=True)
    compare.add_argument("--right", required=True)
    compare.add_argument("--output")
    compare.add_argument("--atol", type=float, default=2e-2)
    compare.add_argument("--rtol", type=float, default=2e-2)
    compare.set_defaults(function=compare_dcp)

    args = parser.parse_args()
    if not math.isfinite(getattr(args, "atol", 0.0)) or not math.isfinite(
        getattr(args, "rtol", 0.0)
    ):
        raise ValueError("Acceptance tolerances must be finite.")
    args.function(args)


if __name__ == "__main__":
    main()
