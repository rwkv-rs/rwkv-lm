import hashlib
import importlib
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

import tomllib

ROOT = Path(__file__).parents[2]
TORCHTITAN_LICENSE_SHA256 = (
    "6eea30995941126beeb99ef775f0968ed8320beb4834ad28d6ae6704a1a92930"
)


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
        "train.py",
        "trainer.py",
    }

    assert not removed_modules.intersection(path.name for path in package.iterdir())
    assert not (ROOT / "train.py").exists()
    assert not list(ROOT.glob("demo-training-run*.sh"))


def test_rwkv7_data_follows_torchtitan_dataset_family_ownership() -> None:
    package = ROOT / "src" / "rwkv_lm"
    model_directory = package / "models" / "rwkv7"
    dataset_directory = package / "rwkv_datasets"

    assert (package / "components" / "tokenizer.py").is_file()
    assert {path.name for path in dataset_directory.glob("*.py")} == {
        "binidx_datasets.py",
        "text_datasets.py",
    }
    assert not (package / "components" / "dataloader.py").exists()
    assert not (package / "datasets" / "binidx.py").exists()
    assert not (package / "datasets" / "dataloader.py").exists()
    assert {path.name for path in model_directory.glob("*.py")} == {
        "__init__.py",
        "config_registry.py",
        "model.py",
        "parallelize.py",
        "sharding.py",
        "state_dict_adapter.py",
    }
    assert not (package / "binidx.py").exists()

    tokenizer_module = importlib.import_module("rwkv_lm.components.tokenizer")
    binidx_module = importlib.import_module("rwkv_lm.rwkv_datasets.binidx_datasets")
    dataloader_module = importlib.import_module("rwkv_lm.rwkv_datasets.text_datasets")
    model_module = importlib.import_module("rwkv_lm.models.rwkv7")
    state_dict_adapter_module = importlib.import_module(
        "rwkv_lm.models.rwkv7.state_dict_adapter"
    )
    assert tokenizer_module.RwkvPretokenizedTokenizer
    assert binidx_module.MMapIndexedDataset
    assert dataloader_module.RwkvDataLoader
    assert not hasattr(model_module, "RwkvPretokenizedTokenizer")
    assert not hasattr(model_module, "RwkvDataLoader")
    assert not hasattr(state_dict_adapter_module, "AdapterCheckpointError")
    assert not hasattr(state_dict_adapter_module, "Rwkv7ArtifactIdentity")
    assert not hasattr(state_dict_adapter_module, "validate_rwkv7_artifact")


def test_tokenizer_import_does_not_load_rwkv7_model_package() -> None:
    script = """
import sys
from rwkv_lm.components.tokenizer import RwkvPretokenizedTokenizer

blocked = {
    "rwkv_lm.models.rwkv7",
    "rwkv_lm.models.rwkv7.model",
    "rwkv_lm.models.rwkv7.parallelize",
    "rwkv_lm.models.rwkv7.state_dict_adapter",
}
loaded = sorted(blocked.intersection(sys.modules))
if loaded:
    raise RuntimeError(f"tokenizer import loaded RWKV-7 model modules: {loaded}")
assert RwkvPretokenizedTokenizer
"""
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )


def test_hosted_cpu_contract_installs_public_fla_and_runs_full_unit_suite() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    workflow = (ROOT / ".github" / "workflows" / "cpu-contract.yml").read_text()

    assert {"pytest>=9.0.0", "ruff>=0.16.0"} <= set(project["dependency-groups"]["dev"])
    assert "--no-install-package flash-rwkv" in workflow
    assert "--no-install-package flash-linear-attention" not in workflow
    assert "pytest -q tests/unit_tests" in workflow
    assert "ruff format --check ." in workflow
    assert "uv lock --check" in workflow


def test_distributions_include_the_torchtitan_bsd_license(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    for filename in ("LICENSE", "README.md", "pyproject.toml"):
        shutil.copy2(ROOT / filename, project / filename)
    shutil.copytree(
        ROOT / "src",
        project / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    output = tmp_path / "dist"
    subprocess.run(
        [
            "uv",
            "build",
            "--no-build-logs",
            "--no-create-gitignore",
            "--out-dir",
            str(output),
            str(project),
        ],
        check=True,
    )

    license_bytes = (ROOT / "LICENSE").read_bytes()
    assert hashlib.sha256(license_bytes).hexdigest() == TORCHTITAN_LICENSE_SHA256

    (source_distribution,) = output.glob("*.tar.gz")
    with tarfile.open(source_distribution, "r:gz") as archive:
        license_members = [
            member
            for member in archive.getmembers()
            if member.isfile() and PurePosixPath(member.name).name == "LICENSE"
        ]
        assert len(license_members) == 1
        extracted_license = archive.extractfile(license_members[0])
        assert extracted_license is not None
        assert extracted_license.read() == license_bytes

    (wheel,) = output.glob("*.whl")
    with zipfile.ZipFile(wheel) as archive:
        license_paths = [
            name for name in archive.namelist() if PurePosixPath(name).name == "LICENSE"
        ]
        assert len(license_paths) == 1
        assert archive.read(license_paths[0]) == license_bytes
