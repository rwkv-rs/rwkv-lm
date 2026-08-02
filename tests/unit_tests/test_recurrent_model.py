from types import MethodType

import pytest
import torch
from torch.nn import functional as F

from rwkv_lm.models.rwkv7 import model as model_module
from rwkv_lm.models.rwkv7 import model_registry
from rwkv_lm.models.rwkv7.config_registry import rwkv7_debugmodel
from rwkv_lm.models.rwkv7.data import RwkvDataLoader, RwkvTokenizer
from rwkv_lm.models.rwkv7.model import (
    Rwkv7Block,
    Rwkv7Model,
    Rwkv7ModelOutput,
    Rwkv7RecurrentState,
)


def _model() -> Rwkv7Model:
    model = model_registry("debugmodel").model.build()
    model.init_states()
    return model


def _fake_block_forward(
    self,
    hidden_states,
    v_first,
    attention_shift,
    wkv_state,
    ffn_shift,
):
    projected = self.att.output(self.att.receptance(hidden_states))
    output = hidden_states + 0.01 * projected
    next_shift = output[:, -1]
    next_wkv = wkv_state + output.float().mean()
    return output, v_first, next_shift, next_wkv, next_shift + ffn_shift * 0.0


def test_real_dataloader_batch_reaches_model_loss_and_backward(monkeypatch) -> None:
    monkeypatch.setattr(Rwkv7Block, "forward", _fake_block_forward)
    model = _model()
    loader = RwkvDataLoader.Config(
        dataset="synthetic",
        vocab_size=1_024,
        infinite=False,
        num_batches=1,
    ).build(
        dp_world_size=1,
        dp_rank=0,
        tokenizer=RwkvTokenizer.Config(vocab_size=1_024).build(tokenizer_path="."),
        seq_len=32,
        local_batch_size=2,
        snapshot_every_n_steps=1,
    )
    inputs, labels = next(iter(loader))
    logits = model(inputs["input"], positions=inputs["positions"])
    loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
    loss.backward()

    assert logits.shape == (2, 32, 1_024)
    assert torch.isfinite(loss)
    assert model.lm_head.weight.grad is not None
    assert torch.count_nonzero(model.lm_head.weight.grad) > 0


def test_explicit_reset_mask_resets_only_selected_recurrent_rows() -> None:
    model = _model()
    seen_states = []

    def fake_forward_chunk(self, hidden_states, state):
        seen_states.append(state.attention_shift.detach().clone())
        increment = hidden_states.shape[1]
        return hidden_states, Rwkv7RecurrentState(
            attention_shift=state.attention_shift + increment,
            wkv=state.wkv + increment,
            ffn_shift=state.ffn_shift + increment,
        )

    model._forward_chunk = MethodType(fake_forward_chunk, model)
    tokens = torch.arange(64).view(2, 32)
    state = model._initial_state(2, torch.float32, torch.device("cpu"))
    state = Rwkv7RecurrentState(
        attention_shift=state.attention_shift + 5,
        wkv=state.wkv + 5,
        ffn_shift=state.ffn_shift + 5,
    )
    output = model(
        tokens,
        positions=torch.arange(32).expand(2, -1),
        state=state,
        reset_mask=torch.tensor([False, True]),
        chunk_size=16,
        return_state=True,
    )

    assert isinstance(output, Rwkv7ModelOutput)
    assert len(seen_states) == 2
    assert torch.all(seen_states[0][:, 0] == 5)
    assert torch.all(seen_states[0][:, 1] == 0)
    assert torch.all(seen_states[1][:, 0] == 21)
    assert torch.all(seen_states[1][:, 1] == 16)
    assert torch.all(output.state.attention_shift[:, 0] == 37)
    assert torch.all(output.state.attention_shift[:, 1] == 32)


def test_continuation_detaches_prior_chunk_state_but_keeps_response_gradient() -> None:
    model = _model()

    def differentiable_chunk(self, hidden_states, state):
        recurrent = state.attention_shift[0]
        output = hidden_states + recurrent[:, None]
        summary = output[:, -1]
        return output, Rwkv7RecurrentState(
            attention_shift=torch.stack((summary, summary)),
            wkv=state.wkv + output.float().mean(),
            ffn_shift=torch.stack((summary, summary)),
        )

    model._forward_chunk = MethodType(differentiable_chunk, model)
    state = model._initial_state(1, torch.float32, torch.device("cpu"))
    state = Rwkv7RecurrentState(
        attention_shift=state.attention_shift.requires_grad_(),
        wkv=state.wkv.requires_grad_(),
        ffn_shift=state.ffn_shift.requires_grad_(),
    )
    tokens = torch.arange(32).remainder(1_024).view(1, 32)
    positions = torch.arange(32, 64).view(1, 32)
    logits = model(
        tokens,
        positions=positions,
        state=state,
        chunk_size=16,
        detach_state_between_chunks=True,
    )
    loss = logits[:, -1].sum()
    loss.backward()

    assert state.attention_shift.grad is not None
    assert torch.count_nonzero(state.attention_shift.grad) == 0
    assert model.tok_embeddings.weight.grad is not None
    assert torch.count_nonzero(model.tok_embeddings.weight.grad) > 0


def test_positions_and_extra_inputs_fail_closed() -> None:
    model = _model()
    tokens = torch.zeros(1, 16, dtype=torch.long)

    with pytest.raises(ValueError, match="positions shape"):
        model(tokens, positions=torch.arange(15).view(1, 15))
    with pytest.raises(TypeError, match="positions must use an integer"):
        model(tokens, positions=torch.arange(16.0).view(1, 16))
    with pytest.raises(TypeError, match="unsupported RWKV model inputs"):
        model(tokens, positions=torch.arange(16).view(1, 16), attention_mask=None)


def test_runtime_provenance_fails_before_model_build(monkeypatch) -> None:
    import transformers.models.rwkv7 as transformers_rwkv7

    monkeypatch.setattr(model_module, "_RWKV7_RUNTIME", None)
    monkeypatch.setattr(
        transformers_rwkv7,
        "validate_rwkv7_runtime_provenance",
        lambda: (_ for _ in ()).throw(RuntimeError("revision mismatch")),
    )
    trainer_config = rwkv7_debugmodel()

    with pytest.raises(RuntimeError, match="revision mismatch"):
        trainer_config.model_spec.model.update_from_config(config=trainer_config)


def test_runtime_preflight_is_once_and_forward_uses_bound_callables(
    monkeypatch,
) -> None:
    import fla.ops.rwkv7 as fla_rwkv7
    import transformers.models.rwkv7 as transformers_rwkv7

    events = []

    def validate_provenance():
        events.append("provenance")
        return {}

    def chunk_rwkv7(*inputs, initial_state, **kwargs):
        del kwargs
        events.append("kernel")
        return inputs[3], initial_state

    monkeypatch.setattr(model_module, "_RWKV7_RUNTIME", None)
    monkeypatch.setattr(
        transformers_rwkv7,
        "validate_rwkv7_runtime_provenance",
        validate_provenance,
    )
    monkeypatch.setattr(fla_rwkv7, "chunk_rwkv7", chunk_rwkv7)
    monkeypatch.setattr(
        fla_rwkv7,
        "get_last_rwkv7_provider",
        lambda: "flash_rwkv",
    )
    trainer_config = rwkv7_debugmodel()
    model_config = trainer_config.model_spec.model
    model_config.update_from_config(config=trainer_config)
    model_config.update_from_config(config=trainer_config)
    model = model_config.build()
    model.init_states()
    time_mix = model.layers["0"].att
    hidden_states = torch.randn(1, 16, model_config.hidden_size)
    shift = torch.zeros(1, model_config.hidden_size)
    num_heads = model_config.hidden_size // model_config.head_size
    wkv = torch.zeros(
        1,
        num_heads,
        model_config.head_size,
        model_config.head_size,
    )
    time_mix(hidden_states, torch.zeros_like(hidden_states), shift, wkv)
    time_mix(hidden_states, torch.zeros_like(hidden_states), shift, wkv)

    assert events == ["provenance", "kernel", "kernel"]


def test_positions_and_bound_runtime_are_fullgraph_compile_compatible(
    monkeypatch,
) -> None:
    def chunk_rwkv7(*inputs, initial_state, **kwargs):
        del kwargs
        return inputs[3], initial_state

    monkeypatch.setattr(
        model_module,
        "_RWKV7_RUNTIME",
        model_module._Rwkv7Runtime(
            chunk_rwkv7=chunk_rwkv7,
            get_last_provider=lambda: "flash_rwkv",
        ),
    )
    model = _model()
    compiled = torch.compile(model, backend="eager", fullgraph=True)
    tokens = torch.arange(16).view(1, 16)
    positions = torch.arange(16).view(1, 16)

    logits = compiled(tokens, positions=positions)

    assert logits.shape == (1, 16, 1_024)
