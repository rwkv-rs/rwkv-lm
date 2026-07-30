import ast
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CUDA_DIR = ROOT / "cuda"
CUDA_LOADERS = (
    ROOT / "src" / "model.py",
    ROOT / "rwkv7_train_simplified.py",
)

spec = importlib.util.spec_from_file_location(
    "rwkv_lm_cuda_sources",
    ROOT / "src" / "cuda_sources.py",
)
assert spec is not None and spec.loader is not None
cuda_sources_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cuda_sources_module)
cuda_sources = cuda_sources_module.cuda_sources


def _declared_cuda_sources(source_file: Path) -> list[tuple[str, ...]]:
    tree = ast.parse(source_file.read_text())
    declared = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "load":
            continue
        sources = next(
            (keyword.value for keyword in node.keywords if keyword.arg == "sources"),
            None,
        )
        assert isinstance(sources, ast.Call)
        assert isinstance(sources.func, ast.Name)
        assert sources.func.id == "cuda_sources"
        declared.append(tuple(ast.literal_eval(argument) for argument in sources.args))
    return declared


def test_all_cuda_extension_sources_are_absolute_and_cwd_independent(
    monkeypatch,
    tmp_path,
):
    declarations = [
        names
        for source_file in CUDA_LOADERS
        for names in _declared_cuda_sources(source_file)
    ]
    assert declarations

    for cwd in (tmp_path, tmp_path / "nested"):
        cwd.mkdir(exist_ok=True)
        monkeypatch.chdir(cwd)
        for names in declarations:
            resolved = cuda_sources(*names)
            expected = [str(CUDA_DIR / name) for name in names]
            assert resolved == expected
            assert all(Path(source).is_absolute() for source in resolved)
            assert all(Path(source).is_file() for source in resolved)
