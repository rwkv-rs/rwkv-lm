from pathlib import Path

import tomllib

ROOT = Path(__file__).parents[2]


def test_torchtitan_entrypoints_and_revision_are_authoritative() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    dependencies = project["project"]["dependencies"]

    assert (
        "flash-linear-attention[cuda,flash-rwkv] @ "
        "git+https://github.com/rwkv-rs/fla-rwkv.git"
        "@88e8ff9d29dcebadb89ebad62ee76951729ea0df"
    ) in dependencies
    assert (
        "flash-rwkv @ git+https://github.com/rwkv-rs/FlashRWKV.git"
        "@c637985558c398de1db6a3c0523b1eec206a88d4"
    ) in dependencies
    assert (
        "torchtitan @ git+https://github.com/pytorch/torchtitan.git"
        "@681fd4b509b183ba33f70d39580b848b36e66ca5"
    ) in dependencies
    assert project["project"]["scripts"]["rwkv-train"] == "torchtitan.train:main"
    assert project["project"]["scripts"]["rwkv-convert-legacy-checkpoint"] == (
        "transformers.models.rwkv7.convert_rwkv7_checkpoint_to_hf:main"
    )


def test_duplicate_generic_training_stack_is_removed() -> None:
    package = ROOT / "src" / "rwkv_lm"
    removed_modules = {
        "activation_checkpointing.py",
        "checkpoint.py",
        "checkpoint_fsdp2.py",
        "checkpoint_runner.py",
        "cli.py",
        "fsdp2_trainer.py",
        "peft.py",
        "trainer.py",
    }

    assert not removed_modules.intersection(path.name for path in package.iterdir())


def test_hosted_cpu_contract_installs_public_fla_and_runs_full_unit_suite() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    workflow = (ROOT / ".github" / "workflows" / "cpu-contract.yml").read_text()

    assert {"pytest>=9.0.0", "ruff>=0.16.0"} <= set(project["dependency-groups"]["dev"])
    assert "--no-install-package flash-rwkv" in workflow
    assert "--no-install-package flash-linear-attention" not in workflow
    assert "pytest -q tests/unit_tests" in workflow
    assert "ruff format --check ." in workflow
    assert "uv lock --check" in workflow
