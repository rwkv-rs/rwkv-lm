"""Narrow compatibility bridges for the pinned TorchTitan/PyTorch stack."""

from __future__ import annotations

from datetime import timedelta

import torch.distributed as dist


def ensure_distributed_set_timeout() -> None:
    """Provide TorchTitan's public timeout helper on older PyTorch releases.

    The pinned TorchTitan calls ``torch.distributed.set_timeout``. PyTorch
    2.13 exposes the same operation on the public ``ProcessGroup`` object but
    not yet as the module-level convenience function.
    """

    if hasattr(dist, "set_timeout"):
        return

    def set_timeout(timeout: timedelta, group: dist.ProcessGroup | None = None) -> None:
        process_group = group if group is not None else dist.group.WORLD
        if process_group is None:
            raise RuntimeError(
                "torch.distributed.set_timeout requires an initialized process group."
            )
        process_group.set_timeout(timeout)

    dist.set_timeout = set_timeout  # type: ignore[attr-defined]


__all__ = ["ensure_distributed_set_timeout"]
