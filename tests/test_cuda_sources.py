import ast
from pathlib import Path

from rwkv_lm.cuda_sources import cuda_sources

PACKAGE_DIR = Path(__file__).resolve().parents[1] / "src" / "rwkv_lm"
CUDA_DIR = PACKAGE_DIR / "cuda"
CUDA_LOADERS = (
    PACKAGE_DIR / "model.py",
    Path(__file__).resolve().parents[1] / "rwkv7_train_simplified.py",
)


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
