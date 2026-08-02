from pathlib import Path

import tomllib

ROOT = Path(__file__).parents[2]


def test_torchtitan_entrypoints_and_revision_are_authoritative() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    dependencies = project["project"]["dependencies"]

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
