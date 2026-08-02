from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed.checkpoint as dcp

from rwkv_lm.components.tokenizer import RwkvPretokenizedTokenizer
from rwkv_lm.rwkv_datasets.binidx_datasets import (
    MMapIndexedDataset,
    data_file_path,
    index_file_path,
)
from rwkv_lm.rwkv_datasets.text_datasets import RwkvDataLoader


def _tokenizer(
    tokenizer_path: Path,
    vocab_size: int = 1_024,
) -> RwkvPretokenizedTokenizer:
    return RwkvPretokenizedTokenizer.Config(vocab_size=vocab_size).build(
        tokenizer_path=str(tokenizer_path)
    )


def _loader(
    config: RwkvDataLoader.Config,
    tokenizer_path: Path,
    *,
    seq_len: int = 16,
    local_batch_size: int = 2,
) -> RwkvDataLoader:
    return config.build(
        dp_world_size=1,
        dp_rank=0,
        tokenizer=_tokenizer(tokenizer_path, config.vocab_size),
        seq_len=seq_len,
        local_batch_size=local_batch_size,
        snapshot_every_n_steps=1,
    )


def _write_binidx(prefix: Path, tokens: np.ndarray) -> None:
    data_file_path(prefix).write_bytes(tokens.tobytes(order="C"))
    with MMapIndexedDataset.Index.writer(
        index_file_path(prefix), tokens.dtype.type
    ) as writer:
        writer.write([len(tokens)], [0])


def _assert_batches_equal(
    expected: tuple[dict[str, torch.Tensor], torch.Tensor],
    observed: tuple[dict[str, torch.Tensor], torch.Tensor],
) -> None:
    expected_inputs, expected_labels = expected
    observed_inputs, observed_labels = observed
    assert expected_inputs.keys() == observed_inputs.keys()
    assert all(
        torch.equal(expected_inputs[name], observed_inputs[name])
        for name in expected_inputs
    )
    assert torch.equal(expected_labels, observed_labels)


def test_synthetic_dataloader_emits_positions_and_resumes_cursor(
    rwkv7_artifact_factory,
) -> None:
    artifact_path, _model_identity = rwkv7_artifact_factory()
    config = RwkvDataLoader.Config(
        dataset="synthetic",
        vocab_size=1_024,
        seed=7,
        infinite=False,
        num_batches=3,
    )
    loader = _loader(config, artifact_path)
    iterator = iter(loader)
    first = next(iterator)
    state = loader.state_dict()
    expected_next = next(iterator)
    resumed = _loader(config, artifact_path)
    resumed.load_state_dict(state)
    observed_next = next(iter(resumed))
    incompatible = _loader(
        RwkvDataLoader.Config(
            dataset="synthetic",
            vocab_size=1_024,
            seed=8,
            infinite=False,
            num_batches=3,
        ),
        artifact_path,
    )

    inputs, labels = first
    assert inputs["input"].shape == labels.shape == (2, 16)
    assert torch.equal(inputs["positions"][0], torch.arange(16))
    _assert_batches_equal(expected_next, observed_next)
    with pytest.raises(ValueError, match="checkpoint identity mismatch"):
        incompatible.load_state_dict(state)


def test_binidx_dataloader_is_stateful_and_requires_real_inputs(
    tmp_path,
    rwkv7_artifact_factory,
) -> None:
    artifact_path, _model_identity = rwkv7_artifact_factory(vocab_size=64)
    prefix = tmp_path / "tokens"
    tokens = np.arange(49, dtype=np.int32)
    _write_binidx(prefix, tokens)
    config = RwkvDataLoader.Config(
        dataset="binidx",
        dataset_path=str(prefix),
        vocab_size=64,
        magic_prime=11,
        infinite=True,
    )
    loader = _loader(config, artifact_path, seq_len=4)
    first_inputs, first_labels = next(iter(loader))

    assert first_inputs["input"].shape == first_labels.shape == (2, 4)
    assert int(first_inputs["input"].max()) < 49

    missing = RwkvDataLoader.Config(dataset="binidx", vocab_size=64)
    with pytest.raises(ValueError, match="--dataloader.dataset-path"):
        _loader(missing, artifact_path, seq_len=4)


def test_dcp_restores_dataloader_cursor(tmp_path, rwkv7_artifact_factory) -> None:
    artifact_path, _model_identity = rwkv7_artifact_factory(vocab_size=128)
    config = RwkvDataLoader.Config(
        dataset="synthetic",
        vocab_size=128,
        seed=11,
        infinite=True,
    )
    loader = _loader(config, artifact_path)
    next(iter(loader))
    checkpoint = tmp_path / "dcp"
    dcp.save({"dataloader": loader}, checkpoint_id=str(checkpoint))
    expected_next = next(iter(loader))

    resumed = _loader(config, artifact_path)
    dcp.load({"dataloader": resumed}, checkpoint_id=str(checkpoint))
    observed_next = next(iter(resumed))

    _assert_batches_equal(expected_next, observed_next)
