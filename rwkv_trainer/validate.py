"""Repository-boundary and dependency validation."""

from __future__ import annotations

import argparse
import ast
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

FORBIDDEN_SUFFIXES = {".cu", ".cuh", ".triton", ".pth"}
FORBIDDEN_NAMES = {"tokenizer.json", "vocab.json", "vocab.txt", "merges.txt"}


def validate_tree(root: Path) -> list[str]:
    violations = []
    for path in root.rglob("*"):
        if (
            any(part in {".git", ".venv", "__pycache__"} for part in path.parts)
            or not path.is_file()
        ):
            continue
        if path.suffix in FORBIDDEN_SUFFIXES or path.name in FORBIDDEN_NAMES:
            violations.append(str(path.relative_to(root)))
        if path.suffix == ".py" and "tests" not in path.parts:
            tree = ast.parse(path.read_text(errors="ignore"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name in {
                    "RwkvTimeMix",
                    "RwkvChannelMix",
                }:
                    violations.append(str(path.relative_to(root)))
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                else:
                    continue
                if any(module == "fla" or module.startswith("fla.") for module in modules):
                    violations.append(str(path.relative_to(root)))
    return sorted(set(violations))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    violations = validate_tree(args.root)
    packages = {}
    for name in ("torchtitan", "transformers", "flashrwkv2", "peft"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    print(json.dumps({"forbidden_files": violations, "packages": packages}, indent=2))
    if violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
