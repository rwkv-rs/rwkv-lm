import json
import shutil
import subprocess
import sys
import venv
import zipfile
from pathlib import Path

import tomllib
from packaging.requirements import Requirement

_STANDARD_RUNTIME_DIRECT_REQUIREMENTS = {
    "flash-linear-attention": (
        frozenset({"cuda"}),
        (
            "git+https://github.com/rwkv-rs/fla-rwkv.git@"
            "1bc262c8c81241e1d339419a31f0aadffa20c210"
        ),
    ),
    "flash-rwkv": (
        frozenset(),
        (
            "git+https://github.com/rwkv-rs/FlashRWKV.git@"
            "866aafd2eed146b0eda1ce03444009ae030f89e3"
        ),
    ),
    "transformers": (
        frozenset(),
        (
            "git+https://github.com/rwkv-rs/transformers-rwkv.git@"
            "eb8248eb9083288e7769518077a1be9c0f7cf7b8"
        ),
    ),
}


def test_standard_runtime_dependencies_use_exact_direct_revisions() -> None:
    project_root = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((project_root / "pyproject.toml").read_text())
    requirements = [Requirement(value) for value in metadata["project"]["dependencies"]]
    direct_requirements = {
        requirement.name: requirement
        for requirement in requirements
        if requirement.url is not None
    }

    assert direct_requirements.keys() == _STANDARD_RUNTIME_DIRECT_REQUIREMENTS.keys()
    for package, (extras, url) in _STANDARD_RUNTIME_DIRECT_REQUIREMENTS.items():
        requirement = direct_requirements[package]
        assert requirement.extras == extras
        assert requirement.url == url
        assert not requirement.specifier
        assert requirement.marker is None


def test_package_import_is_cwd_independent_and_does_not_import_torch(tmp_path):
    code = """
import json
import sys
import rwkv_lm
print(json.dumps({
    "package": rwkv_lm.__name__,
    "torch_loaded": "torch" in sys.modules,
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == {
        "package": "rwkv_lm",
        "torch_loaded": False,
    }


def test_wheel_contains_standard_training_surface_without_native_runtime(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    build_root = tmp_path / "project"
    shutil.copytree(project_root / "src", build_root / "src")
    shutil.copy2(project_root / "README.md", build_root / "README.md")
    shutil.copy2(project_root / "pyproject.toml", build_root / "pyproject.toml")
    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()

    code = f"""
import setuptools.build_meta
setuptools.build_meta.build_wheel({str(wheel_dir)!r})
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=build_root,
        check=True,
        capture_output=True,
        text=True,
    )

    (wheel_path,) = wheel_dir.glob("rwkv_lm-*.whl")
    with zipfile.ZipFile(wheel_path) as wheel:
        names = set(wheel.namelist())
        entry_points_name = next(
            name for name in names if name.endswith(".dist-info/entry_points.txt")
        )
        entry_points = wheel.read(entry_points_name).decode("utf-8")

    assert "rwkv_lm/__init__.py" in names
    assert "rwkv_lm/standard_model.py" in names
    assert "rwkv_lm/fsdp2_trainer.py" in names
    assert "rwkv_lm/infctx.py" in names
    assert "rwkv_lm/peft.py" in names
    assert "rwkv_lm/model.py" not in names
    assert "rwkv_lm/cuda_sources.py" not in names
    assert not any(name.startswith("rwkv_lm/cuda/") for name in names)
    assert not any(name.startswith("src/") for name in names)
    assert "rwkv-train = rwkv_lm.cli:main" in entry_points
    assert (
        "rwkv-convert-legacy-checkpoint = rwkv_lm.standard_model:converter_main"
        in entry_points
    )

    environment = tmp_path / "venv"
    venv.EnvBuilder(with_pip=True).create(environment)
    python = environment / "bin" / "python"
    subprocess.run(
        [python, "-m", "pip", "install", "--no-deps", wheel_path],
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run(
        [
            python,
            "-c",
            (
                "import sys; import rwkv_lm; import rwkv_lm.cli; "
                "assert 'torch' not in sys.modules; print(rwkv_lm.__name__)"
            ),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "rwkv_lm"
