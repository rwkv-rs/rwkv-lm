from pathlib import Path

_CUDA_DIR = Path(__file__).resolve().parent.parent / "cuda"


def cuda_sources(*filenames: str) -> list[str]:
    """Return CUDA extension sources independent of the process working directory."""
    return [str(_CUDA_DIR / filename) for filename in filenames]
