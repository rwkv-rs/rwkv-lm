"""Export a TorchTitan DCP LoRA checkpoint as standard PEFT and optional merged HF artifacts."""

from __future__ import annotations

import argparse
import json
import re
import traceback
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .model import PeftSettings, RwkvModelAdapter
from .provenance import artifact_hashes, dependency_metadata

_BLOCK_KEY = re.compile(r"(\.blocks\.\d+)(\.)")


def _logits(model, tokens: torch.Tensor, *, seed: int) -> torch.Tensor:
    model.train()
    with torch.random.fork_rng(devices=[tokens.device]), torch.no_grad():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        return model(input_ids=tokens, use_cache=False, return_dict=True).logits.float().cpu()


def _inference_logits(model, tokens: torch.Tensor, *, seed: int) -> torch.Tensor:
    model.prepare_for_inference()
    model.eval()
    with torch.random.fork_rng(devices=[tokens.device]), torch.no_grad():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        return model(input_ids=tokens, use_cache=False, return_dict=True).logits.float().cpu()


def _assert_logits_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> dict[str, float | int]:
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    difference = (actual - expected).abs()
    relative = difference / expected.abs().clamp_min(torch.finfo(expected.dtype).tiny)
    return {
        "elements": difference.numel(),
        "max_absolute_difference": difference.max().item(),
        "max_relative_difference": relative.max().item(),
    }


def _require_uniform_floating_dtype(model, dtype: torch.dtype) -> None:
    mismatches = [
        f"parameter {name}={parameter.dtype}"
        for name, parameter in model.named_parameters()
        if parameter.is_floating_point() and parameter.dtype != dtype
    ]
    mismatches.extend(
        f"buffer {name}={buffer.dtype}"
        for name, buffer in model.named_buffers()
        if buffer.is_floating_point() and buffer.dtype != dtype
    )
    if mismatches:
        preview = ", ".join(mismatches[:5])
        raise TypeError(f"RWKV export validation requires uniform {dtype}: {preview}")


def _dcp_model_state(
    model_state: dict[str, torch.Tensor], source_keys: set[str]
) -> dict[str, torch.Tensor]:
    result = {}
    missing = []
    for key, tensor in model_state.items():
        wrapped_key = _BLOCK_KEY.sub(r"\1._checkpoint_wrapped_module\2", key, count=1)
        candidates = [
            candidate for candidate in dict.fromkeys((key, wrapped_key)) if candidate in source_keys
        ]
        if len(candidates) != 1:
            missing.append(key)
            continue
        result[candidates[0]] = tensor
    if missing:
        preview = ", ".join(missing[:5])
        raise KeyError(
            f"DCP does not contain a unique canonical model key for {len(missing)} parameters: "
            f"{preview}"
        )
    return result


def _export(args: argparse.Namespace, output: Path) -> None:
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
    model_state = adapter.state_dict()
    reader = dcp.FileSystemReader(args.checkpoint)
    source_keys = set(reader.read_metadata().state_dict_metadata)
    checkpoint_state = _dcp_model_state(model_state, source_keys)
    dcp.load(checkpoint_state, storage_reader=reader)
    adapter.load_state_dict(model_state, strict=True)
    # DCP preserves the base artifact's storage dtype (for this checkpoint,
    # FP16) while LoRA tensors follow the BF16 training policy. FSDP normally
    # normalizes both for compute; the standalone exporter must do so itself.
    adapter.to(device=args.device, dtype=torch.bfloat16)
    _require_uniform_floating_dtype(adapter, torch.bfloat16)
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
        AutoModelForCausalLM.from_pretrained(
            args.base_model,
            local_files_only=True,
            dtype=torch.bfloat16,
        ),
        adapter_dir,
        local_files_only=True,
    ).to(device=args.device, dtype=torch.bfloat16)
    _require_uniform_floating_dtype(reloaded, torch.bfloat16)
    adapter_training_validation = _assert_logits_close(
        _logits(reloaded, tokens, seed=args.seed),
        expected,
        atol=args.atol,
        rtol=args.rtol,
    )

    merged_inference_validation = None
    if args.merge:
        reloaded.to(device=args.device, dtype=torch.float16)
        _require_uniform_floating_dtype(reloaded, torch.float16)
        unmerged_inference = _inference_logits(reloaded, tokens, seed=args.seed)

        # Merge in FP32 so LoRA deltas smaller than a BF16 base-weight ULP are
        # not discarded. The canonical inference provider consumes FP16 model
        # weights (with BF16 embeddings), so store the standalone artifact in
        # FP16 and validate it through that public path.
        merge_source = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(
                args.base_model,
                local_files_only=True,
                dtype=torch.float32,
            ),
            adapter_dir,
            local_files_only=True,
        )
        _require_uniform_floating_dtype(merge_source, torch.float32)
        merged = merge_source.merge_and_unload().to(dtype=torch.float16)
        merged_dir = output / "merged"
        merged.save_pretrained(merged_dir, safe_serialization=True)
        AutoTokenizer.from_pretrained(args.base_model, local_files_only=True).save_pretrained(
            merged_dir
        )
        reloaded_merged = AutoModelForCausalLM.from_pretrained(
            merged_dir,
            local_files_only=True,
            dtype=torch.float16,
        ).to(device=args.device)
        merged_inference_validation = _assert_logits_close(
            _inference_logits(reloaded_merged, tokens, seed=args.seed),
            unmerged_inference,
            atol=args.atol,
            rtol=args.rtol,
        )

    training_summary = None
    if args.training_config:
        training_summary = json.loads(Path(args.training_config).read_text())
    metadata = {
        **dependency_metadata(),
        "trainer_revision": args.trainer_revision,
        "base_model": args.base_model,
        "base_revision": args.base_revision,
        "base_artifact_sha256": artifact_hashes(args.base_model),
        "peft": {
            "rank": args.rank,
            "alpha": args.alpha,
            "dropout": args.dropout,
            "target_modules": args.target_modules,
        },
        "training_config": training_summary,
        "validation": {
            "seed": args.seed,
            "atol": args.atol,
            "rtol": args.rtol,
            "adapter_training": {
                "compute_dtype": "bfloat16",
                "reference": "dcp_training_model",
                "candidate": "adapter_reload",
                **adapter_training_validation,
            },
            "merged_inference": None
            if merged_inference_validation is None
            else {
                "compute_dtype": "float16",
                "reference": "unmerged_peft_prepare_for_inference",
                "candidate": "merged_reload_prepare_for_inference",
                "storage_dtype": "float16",
                **merged_inference_validation,
            },
        },
        "artifacts": {
            "adapter_sha256": artifact_hashes(adapter_dir),
            "merged_sha256": None if not args.merge else artifact_hashes(output / "merged"),
        },
    }
    (output / "rwkv_training_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def export(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError(
            "RWKV artifact logits validation requires CUDA and FlashRWKV2; no fallback is provided."
        )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        _export(args, output)
    except BaseException as error:
        failure = {
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        (output / "_FAILED.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--trainer-revision", required=True)
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
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    export(parser.parse_args())


if __name__ == "__main__":
    main()
