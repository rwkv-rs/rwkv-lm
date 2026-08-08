"""Console entry point preserving TorchTitan's module/config launch contract."""

from __future__ import annotations

import os
import sys


def train() -> None:
    from torchtitan.train import main

    module = os.environ.get("RWKV_TORCHTITAN_MODULE", "rwkv_trainer")
    config = os.environ.get("RWKV_TORCHTITAN_CONFIG", "rwkv7_debug")
    sys.argv[1:1] = ["--module", module, "--config", config]
    main()


__all__ = ["train"]
