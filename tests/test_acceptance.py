from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp

from rwkv_trainer.binidx import RwkvDataLoader
from tools.rwkv_acceptance import compare_dcp, make_binidx


def test_acceptance_fixture_is_canonical_binidx(tmp_path: Path, capsys) -> None:
    prefix = tmp_path / "acceptance"
    make_binidx(
        Namespace(
            output_prefix=str(prefix),
            vocab_size=256,
            seq_len=4096,
            slots=1024,
            magic_prime=1019,
            seed=20260808,
        )
    )
    report = json.loads(capsys.readouterr().out)
    assert report["tokens"] == 4_194_305
    assert report["magic_prime_ratio"] == pytest.approx(1019 / 1024)
    loader = RwkvDataLoader.Config(
        dataset="binidx",
        dataset_path=str(prefix),
        vocab_size=256,
        magic_prime=1019,
    ).build(
        dp_world_size=1,
        dp_rank=0,
        tokenizer=object(),
        seq_len=4096,
        local_batch_size=1,
    )
    tokens = next(iter(loader))[0]["input"]
    assert tokens.shape == (1, 4096)
    assert torch.all((0 <= tokens) & (tokens < 256))


def test_acceptance_dcp_comparison_streams_tensor_and_bytes(tmp_path: Path, capsys) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    state = {
        "model": {
            "bf16": torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
            "fp32": torch.tensor([3.0], dtype=torch.float32),
        },
        "trainer": {"step": 5, "cursor": 10},
    }
    dcp.save(state, checkpoint_id=str(left))
    dcp.save(state, checkpoint_id=str(right))
    arguments = Namespace(left=str(left), right=str(right), output=None, atol=2e-2, rtol=2e-2)
    compare_dcp(arguments)
    report = json.loads(capsys.readouterr().out)
    assert report["differences"] == []
    assert report["keys_compared"] == 4

    changed = {
        "model": {
            "bf16": torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
            "fp32": torch.tensor([4.0], dtype=torch.float32),
        },
        "trainer": {"step": 5, "cursor": 10},
    }
    mismatch = tmp_path / "mismatch"
    dcp.save(changed, checkpoint_id=str(mismatch))
    arguments.right = str(mismatch)
    with pytest.raises(SystemExit):
        compare_dcp(arguments)
    mismatch_report = json.loads(capsys.readouterr().out)
    assert mismatch_report["differences"][0]["reason"] == "bitwise tensor mismatch"
