import json
import shutil
import subprocess
import sys
import venv
import zipfile
from pathlib import Path


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
