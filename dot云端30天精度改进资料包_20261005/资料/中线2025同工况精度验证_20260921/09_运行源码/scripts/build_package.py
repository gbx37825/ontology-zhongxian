"""Build a source-only standalone archive from an explicit allowlist."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = {"wxq_pipedream", "pipedream_solver", "scripts", "tests"}
DATA_FILES = {"examples/示例模型.json", "examples/自适应预热.json",
              "baseline/交付验证摘要.json", "baseline/基线清单.json", "baseline/完整库源码清单.json",
              "baseline/完整库验证.json", "baseline/最新基线完整验证.json", "baseline/独立包快速验证.json"}
ROOT_FILES = {"pyproject.toml", "requirements.txt", "uv.lock", "run.sh", "Dockerfile",
              ".dockerignore", ".gitignore", "LICENSE-GPL-3.0.txt",
              "使用说明.md", "完整库说明.md", "第三方组件说明.md", "通用预热设计.md", "Numba默认后端说明.md"}


def source_files():
    files = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(ROOT)
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".nbi", ".nbc"}:
            continue
        allowed_source = relative.parts[0] in DIRECTORIES and path.suffix == ".py"
        if (len(relative.parts) == 1 and path.name in ROOT_FILES) or allowed_source or str(relative) in DATA_FILES or str(relative) == "pipedream_solver/LICENSE":
            if path.suffix == ".py":
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    names = [node.module or ""] if isinstance(node, ast.ImportFrom) else [n.name for n in node.names] if isinstance(node, ast.Import) else []
                    if any(name == "src" or name.startswith(("src.", "hydros_")) for name in names):
                        raise ValueError(f"Platform import in package: {relative}")
            files.append(path)
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "dist/WXQ_PipedreamNumba默认独立包_1.3.0.zip")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("交付包已存在，请另选输出名称，不能覆盖历史包")
    files = source_files()
    entries = [{"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in files]
    manifest = {"version": "1.3.0", "default_backend": "numba", "file_count_in_archive": len(files) + 2,
                "files": entries, "historical_validation_directory": "baseline",
                "excluded": ["platform services", "real models", "observed tables", "local validation runs", "virtualenv", "compiled caches"]}
    manifest_path = ROOT / "PACKAGE_CONTENTS.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    files.append(manifest_path)
    checksums = ROOT / "SHA256SUMS"
    checksums.write_text("".join(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(ROOT)}\n" for path in files), encoding="utf-8")
    files.append(checksums)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, f"wxq-pipedream/{path.relative_to(ROOT)}")
    with zipfile.ZipFile(args.output) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(files)
    print(json.dumps({"archive": str(args.output), "files": len(files),
                      "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
