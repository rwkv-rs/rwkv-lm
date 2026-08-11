"""Repository-boundary and dependency validation."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

from .provenance import (
    FLASHRWKV2_VERSION,
    PEFT_VERSION,
    TOKENIZERS_OID,
    TORCHTITAN_OID,
    TRANSFORMERS_OID,
)

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
    parser.add_argument("--require-locked-dependencies", action="store_true")
    args = parser.parse_args()
    violations = validate_tree(args.root)
    packages = {}
    for name in ("torchtitan", "transformers", "tokenizers", "flashrwkv2", "peft"):
        try:
            installed = distribution(name)
            direct_url = installed.read_text("direct_url.json")
            module = importlib.util.find_spec(name)
            packages[name] = {
                "version": installed.version,
                "source": None if module is None else module.origin,
                "direct_url": None if direct_url is None else json.loads(direct_url),
            }
        except PackageNotFoundError:
            packages[name] = None
    dependency_violations = []
    if args.require_locked_dependencies:
        expected_versions = {"flashrwkv2": FLASHRWKV2_VERSION, "peft": PEFT_VERSION}
        expected_oids = {
            "torchtitan": TORCHTITAN_OID,
            "transformers": TRANSFORMERS_OID,
            "tokenizers": TOKENIZERS_OID,
        }
        for name, expected in expected_versions.items():
            installed = packages[name]
            actual = None if installed is None else installed["version"]
            if actual != expected:
                dependency_violations.append(
                    f"{name} version mismatch: expected {expected}, got {actual}"
                )
        for name, expected in expected_oids.items():
            installed = packages[name]
            direct_url = None if installed is None else installed["direct_url"]
            vcs_info = None if direct_url is None else direct_url.get("vcs_info")
            actual = None if vcs_info is None else vcs_info.get("commit_id")
            if actual != expected:
                dependency_violations.append(
                    f"{name} revision mismatch: expected {expected}, got {actual}"
                )
    print(
        json.dumps(
            {
                "forbidden_files": violations,
                "dependency_violations": dependency_violations,
                "packages": packages,
            },
            indent=2,
        )
    )
    if violations or dependency_violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
