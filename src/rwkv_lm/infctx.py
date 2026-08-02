"""Explicit recurrent-state contract for truncated-BPTT infctx execution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import torch


class InfctxContractError(ValueError):
    """Raised when an infctx request cannot preserve recurrent semantics."""


class InfctxBoundary(str, Enum):
    """State ownership at the start of one infctx call."""

    RESET = "reset"
    CONTINUE = "continue"


@dataclass(frozen=True)
class InfctxState:
    """RWKV recurrent state carried between real backend invocations."""

    shift_states: torch.Tensor
    wkv_states: torch.Tensor
    tokens_seen: int

    def __post_init__(self) -> None:
        if not isinstance(self.shift_states, torch.Tensor) or not isinstance(
            self.wkv_states,
            torch.Tensor,
        ):
            raise InfctxContractError("infctx state values must be Torch tensors")
        if (
            isinstance(self.tokens_seen, bool)
            or not isinstance(self.tokens_seen, int)
            or self.tokens_seen < 0
        ):
            raise InfctxContractError(
                "infctx state tokens_seen must be a non-negative integer"
            )

    def detached(self) -> InfctxState:
        """Drop cross-boundary autograd history without changing state values."""

        return InfctxState(
            shift_states=self.shift_states.detach(),
            wkv_states=self.wkv_states.detach(),
            tokens_seen=self.tokens_seen,
        )


@dataclass(frozen=True)
class InfctxResult:
    """Differentiable token outputs plus their recurrent continuation state.

    ``recurrent_chunk_forward`` detaches the state before returning it to a
    caller. Backend chunk functions use the same value type while the state is
    still attached, immediately before that explicit truncation boundary.
    """

    output: torch.Tensor
    state: InfctxState

    def __post_init__(self) -> None:
        if not isinstance(self.output, torch.Tensor):
            raise InfctxContractError("infctx output must be a Torch tensor")
        if not isinstance(self.state, InfctxState):
            raise InfctxContractError("infctx result state must be InfctxState")


def validate_infctx_chunk_ctx(
    chunk_ctx: int,
    *,
    ctx_len: int,
    kernel_chunk_len: int = 16,
) -> int:
    """Validate the public chunk boundary required by the recurrent backend."""

    for value, name in (
        (chunk_ctx, "chunk_ctx"),
        (ctx_len, "ctx_len"),
        (kernel_chunk_len, "kernel_chunk_len"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise InfctxContractError(f"infctx {name} must be a positive integer")
    if chunk_ctx >= ctx_len:
        raise InfctxContractError("infctx chunk_ctx must be smaller than ctx_len")
    if ctx_len % kernel_chunk_len != 0:
        raise InfctxContractError(
            "infctx ctx_len must be divisible by the backend kernel chunk length"
        )
    if chunk_ctx % kernel_chunk_len != 0:
        raise InfctxContractError(
            "infctx chunk_ctx must be divisible by the backend kernel chunk length"
        )
    return chunk_ctx


def recurrent_chunk_forward(
    input_ids: torch.Tensor,
    *,
    chunk_ctx: int,
    ctx_len: int,
    boundary: InfctxBoundary | str,
    state: InfctxState | None,
    reset_state: Callable[[], InfctxState],
    validate_state: Callable[[InfctxState], None],
    forward_chunk: Callable[[torch.Tensor, InfctxState], InfctxResult],
    kernel_chunk_len: int = 16,
) -> InfctxResult:
    """Run real recurrent chunks with a detached state at every boundary.

    ``RESET`` owns a new zero state for the whole batch. ``CONTINUE`` requires
    a previously returned state. Per-row resets and packed mixed boundaries are
    deliberately outside this contract; callers must form homogeneous batches.
    Token outputs are never detached, so losses on every response chunk retain
    their local parameter gradients while recurrent history is truncated. The
    provider validator sees the detached state that will cross each boundary,
    including the final state returned to the caller.
    """

    chunk_ctx = validate_infctx_chunk_ctx(
        chunk_ctx,
        ctx_len=ctx_len,
        kernel_chunk_len=kernel_chunk_len,
    )
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise InfctxContractError("infctx input_ids must have shape [B, T]")
    sequence_length = input_ids.shape[1]
    if sequence_length <= 0:
        raise InfctxContractError("infctx input sequence must not be empty")
    if sequence_length > ctx_len:
        raise InfctxContractError("infctx input sequence exceeds ctx_len")
    if sequence_length % kernel_chunk_len != 0:
        raise InfctxContractError(
            "infctx input length must be divisible by the backend kernel chunk "
            "length; silent padding would corrupt the returned recurrent state"
        )
    try:
        normalized_boundary = InfctxBoundary(boundary)
    except ValueError as error:
        raise InfctxContractError(
            "infctx boundary must be 'reset' or 'continue'"
        ) from error

    if normalized_boundary is InfctxBoundary.RESET:
        if state is not None:
            raise InfctxContractError("infctx reset boundary does not accept state")
        current_state = reset_state()
    else:
        if state is None:
            raise InfctxContractError("infctx continue boundary requires state")
        current_state = state
    if not isinstance(current_state, InfctxState):
        raise InfctxContractError("infctx state factory must return InfctxState")
    if normalized_boundary is InfctxBoundary.RESET and current_state.tokens_seen != 0:
        raise InfctxContractError(
            "infctx reset state must start with tokens_seen equal to zero"
        )
    current_state = current_state.detached()
    validate_state(current_state)

    outputs = []
    for input_chunk in input_ids.split(chunk_ctx, dim=1):
        result = forward_chunk(input_chunk, current_state)
        if not isinstance(result, InfctxResult):
            raise InfctxContractError(
                "infctx backend forward_chunk must return InfctxResult"
            )
        if result.output.ndim < 2 or result.output.shape[:2] != input_chunk.shape:
            raise InfctxContractError(
                "infctx backend output must preserve the input [B, T] axes"
            )
        expected_tokens_seen = current_state.tokens_seen + input_chunk.shape[1]
        if result.state.tokens_seen != expected_tokens_seen:
            raise InfctxContractError(
                "infctx backend state tokens_seen did not advance by chunk length"
            )
        next_state = result.state.detached()
        validate_state(next_state)
        outputs.append(result.output)
        current_state = next_state

    return InfctxResult(
        output=torch.cat(outputs, dim=1),
        state=current_state,
    )


__all__ = [
    "InfctxBoundary",
    "InfctxContractError",
    "InfctxResult",
    "InfctxState",
    "recurrent_chunk_forward",
    "validate_infctx_chunk_ctx",
]
