import importlib.resources
import json
import multiprocessing
import pickle
import shutil
import subprocess
import sys
import venv
import zipfile
from pathlib import Path

import rwkv_lm
from rwkv_lm.cuda_sources import cuda_sources


def _inspect_import(queue) -> None:
    import rwkv_lm.cuda_sources

    function = rwkv_lm.cuda_sources.cuda_sources
    queue.put((rwkv_lm.cuda_sources.__name__, function.__module__))


def test_package_import_is_cwd_independent_and_does_not_import_torch(tmp_path):
    code = """
import json
import sys
import rwkv_lm
from rwkv_lm.cuda_sources import cuda_sources
print(json.dumps({
    "package": rwkv_lm.__name__,
    "function_module": cuda_sources.__module__,
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
        "function_module": "rwkv_lm.cuda_sources",
        "torch_loaded": False,
    }


def test_import_identity_survives_pickle_and_spawn():
    restored = pickle.loads(pickle.dumps(cuda_sources))
    assert restored is cuda_sources
    assert restored.__module__ == "rwkv_lm.cuda_sources"

    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_inspect_import, args=(queue,))
    process.start()
    process.join(timeout=30)

    assert process.exitcode == 0
    assert queue.get(timeout=5) == (
        "rwkv_lm.cuda_sources",
        "rwkv_lm.cuda_sources",
    )


def test_cuda_sources_are_package_resources():
    cuda_dir = importlib.resources.files(rwkv_lm).joinpath("cuda")
    packaged_sources = {
        resource.name
        for resource in cuda_dir.iterdir()
        if resource.name.endswith((".cpp", ".cu"))
    }
    resolved_sources = {
        Path(source).name
        for source in cuda_sources(*sorted(packaged_sources))
    }

    assert packaged_sources
    assert resolved_sources == packaged_sources


def test_wheel_contains_package_and_cuda_assets(tmp_path):
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
    source_names = {
        path.name
        for path in (project_root / "src" / "rwkv_lm" / "cuda").iterdir()
    }
    wheel_sources = {
        Path(name).name
        for name in names
        if name.startswith("rwkv_lm/cuda/")
    }

    assert "rwkv_lm/__init__.py" in names
    assert wheel_sources == source_names
    assert not any(name.startswith("src/") for name in names)

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
                "from pathlib import Path; import rwkv_lm; "
                "from rwkv_lm.cuda_sources import cuda_sources; "
                "assert Path(cuda_sources('wkv7_op.cpp')[0]).is_file(); "
                "print(rwkv_lm.__name__)"
            ),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "rwkv_lm"
