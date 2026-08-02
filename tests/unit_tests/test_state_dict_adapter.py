import pytest
import torch

from rwkv_lm.models.rwkv7.model import Rwkv7Model
from rwkv_lm.models.rwkv7.state_dict_adapter import Rwkv7StateDictAdapter


def test_transformers_state_dict_names_round_trip() -> None:
    adapter = Rwkv7StateDictAdapter(Rwkv7Model.Config(), None)
    native = {
        "tok_embeddings.weight": torch.tensor([1.0]),
        "layers.0.att.receptance.weight": torch.tensor([2.0]),
        "norm.weight": torch.tensor([3.0]),
        "lm_head.weight": torch.tensor([4.0]),
    }

    hf = adapter.to_hf(native)

    assert set(hf) == {
        "model.embeddings.weight",
        "model.blocks.0.att.receptance.weight",
        "model.ln_out.weight",
        "head.weight",
    }
    assert adapter.from_hf(hf) == native


def test_unknown_state_dict_name_fails_closed() -> None:
    adapter = Rwkv7StateDictAdapter(Rwkv7Model.Config(), None)

    with pytest.raises(KeyError, match="unrecognized RWKV-7"):
        adapter.to_hf({"native_fallback.weight": torch.tensor([1.0])})
