from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from rwkv_lm.infctx import (
    InfctxBoundary,
    InfctxContractError,
    InfctxResult,
    InfctxState,
    recurrent_chunk_forward,
    validate_infctx_chunk_ctx,
)


class _TinyRecurrentBackend(nn.Module):
    """CPU fixture with value-sensitive recurrent state and shared parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.input_scale = nn.Parameter(torch.tensor(0.7))
        self.readout_scale = nn.Parameter(torch.tensor(1.3))
        self.incoming_state_requires_grad = []

    @staticmethod
    def reset(batch_size: int) -> InfctxState:
        return InfctxState(
            shift_states=torch.zeros(1, 2, batch_size, 1),
            wkv_states=torch.zeros(1, batch_size, 1, 1, 1),
            tokens_seen=0,
        )

    def forward_chunk(
        self,
        values: torch.Tensor,
        state: InfctxState,
    ) -> InfctxResult:
        self.incoming_state_requires_grad.append(
            state.shift_states.requires_grad or state.wkv_states.requires_grad
        )
        hidden = state.wkv_states.reshape(values.shape[0], 1)
        outputs = []
        for token in values.unbind(dim=1):
            hidden = torch.tanh(0.5 * hidden + self.input_scale * token[:, None])
            outputs.append(self.readout_scale * hidden)
        output = torch.stack(outputs, dim=1)
        return InfctxResult(
            output=output,
            state=InfctxState(
                shift_states=torch.stack((hidden, hidden), dim=0).unsqueeze(0),
                wkv_states=hidden.reshape(1, values.shape[0], 1, 1, 1),
                tokens_seen=state.tokens_seen + values.shape[1],
            ),
        )


def _run(
    backend: _TinyRecurrentBackend,
    values: torch.Tensor,
    *,
    boundary: InfctxBoundary,
    state: InfctxState | None = None,
) -> InfctxResult:
    return recurrent_chunk_forward(
        values,
        chunk_ctx=16,
        ctx_len=64,
        boundary=boundary,
        state=state,
        reset_state=lambda: backend.reset(values.shape[0]),
        forward_chunk=backend.forward_chunk,
    )


def test_recurrent_chunks_match_one_call_values_and_keep_token_outputs_live() -> None:
    values = torch.linspace(-1, 1, 32).reshape(1, 32)
    one_call_backend = _TinyRecurrentBackend()
    chunk_backend = copy.deepcopy(one_call_backend)

    one_call = one_call_backend.forward_chunk(
        values,
        one_call_backend.reset(values.shape[0]),
    )
    chunked = _run(
        chunk_backend,
        values,
        boundary=InfctxBoundary.RESET,
    )

    torch.testing.assert_close(chunked.output, one_call.output, rtol=0, atol=0)
    torch.testing.assert_close(
        chunked.state.shift_states,
        one_call.state.shift_states,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        chunked.state.wkv_states,
        one_call.state.wkv_states,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        chunked.output.square().mean(),
        one_call.output.square().mean(),
        rtol=0,
        atol=0,
    )
    assert chunk_backend.incoming_state_requires_grad == [False, False]
    assert chunked.output.requires_grad
    assert not chunked.state.shift_states.requires_grad
    assert not chunked.state.wkv_states.requires_grad


def test_response_gradient_matches_one_call_from_detached_prefix_state() -> None:
    values = torch.linspace(-0.75, 0.75, 32).reshape(1, 32)
    chunk_backend = _TinyRecurrentBackend()
    reference_backend = copy.deepcopy(chunk_backend)

    chunked = _run(
        chunk_backend,
        values,
        boundary=InfctxBoundary.RESET,
    )
    chunk_loss = chunked.output[:, 16:].square().mean()

    prefix = reference_backend.forward_chunk(
        values[:, :16],
        reference_backend.reset(values.shape[0]),
    )
    response_reference = reference_backend.forward_chunk(
        values[:, 16:],
        prefix.state.detached(),
    )
    reference_loss = response_reference.output.square().mean()

    torch.testing.assert_close(
        chunked.output[:, 16:],
        response_reference.output,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(chunk_loss, reference_loss, rtol=0, atol=0)
    chunk_loss.backward()
    reference_loss.backward()
    for chunk_parameter, reference_parameter in zip(
        chunk_backend.parameters(),
        reference_backend.parameters(),
        strict=True,
    ):
        assert chunk_parameter.grad is not None
        assert torch.count_nonzero(chunk_parameter.grad) > 0
        torch.testing.assert_close(
            chunk_parameter.grad,
            reference_parameter.grad,
            rtol=0,
            atol=0,
        )


def test_continue_carries_state_while_reset_starts_a_new_sequence() -> None:
    first_values = torch.linspace(-1, 0, 16).reshape(1, 16)
    second_values = torch.linspace(0.1, 1, 16).reshape(1, 16)
    backend = _TinyRecurrentBackend()

    first = _run(backend, first_values, boundary=InfctxBoundary.RESET)
    continued = _run(
        backend,
        second_values,
        boundary=InfctxBoundary.CONTINUE,
        state=first.state,
    )
    reset = _run(backend, second_values, boundary=InfctxBoundary.RESET)
    one_call = backend.forward_chunk(
        torch.cat((first_values, second_values), dim=1),
        backend.reset(first_values.shape[0]),
    )

    torch.testing.assert_close(
        torch.cat((first.output, continued.output), dim=1),
        one_call.output,
        rtol=0,
        atol=0,
    )
    assert continued.state.tokens_seen == 32
    assert reset.state.tokens_seen == 16
    assert not torch.equal(continued.output, reset.output)


@pytest.mark.parametrize(
    ("chunk_ctx", "ctx_len", "message"),
    [
        (0, 32, "positive integer"),
        (32, 32, "smaller than ctx_len"),
        (8, 32, "divisible"),
    ],
)
def test_chunk_context_rejects_unsafe_boundaries(
    chunk_ctx: int,
    ctx_len: int,
    message: str,
) -> None:
    with pytest.raises(InfctxContractError, match=message):
        validate_infctx_chunk_ctx(chunk_ctx, ctx_len=ctx_len)


def test_recurrent_contract_rejects_ambiguous_state_and_silent_padding() -> None:
    backend = _TinyRecurrentBackend()
    aligned = torch.zeros(1, 16)
    state = backend.reset(1)

    with pytest.raises(InfctxContractError, match="reset boundary does not accept"):
        _run(
            backend,
            aligned,
            boundary=InfctxBoundary.RESET,
            state=state,
        )
    with pytest.raises(InfctxContractError, match="continue boundary requires"):
        _run(backend, aligned, boundary=InfctxBoundary.CONTINUE)
    with pytest.raises(InfctxContractError, match="tokens_seen equal to zero"):
        recurrent_chunk_forward(
            aligned,
            chunk_ctx=16,
            ctx_len=32,
            boundary=InfctxBoundary.RESET,
            state=None,
            reset_state=lambda: InfctxState(
                shift_states=state.shift_states,
                wkv_states=state.wkv_states,
                tokens_seen=1,
            ),
            forward_chunk=backend.forward_chunk,
        )
    with pytest.raises(InfctxContractError, match="silent padding"):
        recurrent_chunk_forward(
            torch.zeros(1, 17),
            chunk_ctx=16,
            ctx_len=32,
            boundary=InfctxBoundary.RESET,
            state=None,
            reset_state=lambda: backend.reset(1),
            forward_chunk=backend.forward_chunk,
        )
