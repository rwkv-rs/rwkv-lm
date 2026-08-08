"""Export a TorchTitan DCP LoRA checkpoint as standard PEFT and optional merged HF artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .model import PeftSettings, RwkvModelAdapter
from .provenance import dependency_metadata


def _logits(model, tokens: torch.Tensor, *, seed: int) -> torch.Tensor:
    model.train()
    with torch.random.fork_rng(devices=[tokens.device]), torch.no_grad():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        return model(input_ids=tokens, use_cache=False, return_dict=True).logits.float().cpu()


def export(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError(
            "RWKV artifact logits validation requires CUDA and FlashRWKV2; no fallback is provided."
        )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    settings = PeftSettings(
        enabled=True,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        target_modules=args.target_modules,
    )
    adapter = RwkvModelAdapter(
        RwkvModelAdapter.Config(hf_assets_path=args.base_model, peft=settings)
    ).to(device=args.device, dtype=torch.bfloat16)
    state = adapter.state_dict()
    dcp.load(state, checkpoint_id=args.checkpoint)
    adapter.load_state_dict(state, strict=False)
    if not isinstance(adapter.hf_model, PeftModel):
        raise TypeError("Expected PEFT-wrapped model during adapter export.")

    tokens = torch.randint(
        0,
        adapter.rwkv_model.config.vocab_size,
        (1, 16),
        generator=torch.Generator(device=args.device).manual_seed(args.seed),
        device=args.device,
    )
    expected = _logits(adapter.hf_model, tokens, seed=args.seed)
    adapter_dir = output / "adapter"
    adapter.hf_model.save_pretrained(adapter_dir, safe_serialization=True)
    reloaded = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(args.base_model),
        adapter_dir,
    ).to(device=args.device, dtype=torch.bfloat16)
    torch.testing.assert_close(
        _logits(reloaded, tokens, seed=args.seed), expected, atol=args.atol, rtol=args.rtol
    )

    if args.merge:
        merged = reloaded.merge_and_unload()
        merged_dir = output / "merged"
        merged.save_pretrained(merged_dir, safe_serialization=True)
        AutoTokenizer.from_pretrained(args.base_model).save_pretrained(merged_dir)
        reloaded_merged = (
            type(merged).from_pretrained(merged_dir).to(device=args.device, dtype=torch.bfloat16)
        )
        torch.testing.assert_close(
            _logits(reloaded_merged, tokens, seed=args.seed),
            expected,
            atol=args.atol,
            rtol=args.rtol,
        )

    training_summary = None
    if args.training_config:
        training_summary = json.loads(Path(args.training_config).read_text())
    metadata = {
        **dependency_metadata(),
        "base_model": args.base_model,
        "base_revision": args.base_revision,
        "peft": {
            "rank": args.rank,
            "alpha": args.alpha,
            "dropout": args.dropout,
            "target_modules": args.target_modules,
        },
        "training_config": training_summary,
        "validation": {"seed": args.seed, "atol": args.atol, "rtol": args.rtol},
    }
    (output / "rwkv_training_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--training-config")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=["receptance", "key", "value", "output"],
    )
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--atol", type=float, default=4e-2)
    parser.add_argument("--rtol", type=float, default=4e-2)
    export(parser.parse_args())


if __name__ == "__main__":
    main()
