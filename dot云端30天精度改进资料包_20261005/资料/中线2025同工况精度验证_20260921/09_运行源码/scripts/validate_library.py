from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import unittest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MODULES = (
    "geometry", "hydraulics", "hydrology", "infiltration", "ngeometry",
    "ninfiltration", "nquality", "nsuperlink", "nutils", "simulation",
    "storage", "superlink", "transport", "utils", "visualization",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Import and smoke-test the complete Pipedream library")
    parser.add_argument("--output", type=Path, default=PACKAGE_ROOT / "outputs" / "library_validation")
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(PACKAGE_ROOT))
    actual_modules = {path.stem for path in (PACKAGE_ROOT / "pipedream_solver").glob("*.py")
                      if path.stem != "__init__"}
    if actual_modules != set(MODULES):
        raise AssertionError(f"Library inventory mismatch: {sorted(actual_modules)}")
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pipedream-library-") as cache:
        os.environ["MPLBACKEND"] = "Agg"
        os.environ["MPLCONFIGDIR"] = str(Path(cache) / "matplotlib")
        os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(cache) / "numba"))
        os.environ["XDG_CACHE_HOME"] = str(Path(cache) / "xdg")
        imports = []
        for name in MODULES:
            module = importlib.import_module(f"pipedream_solver.{name}")
            path = Path(module.__file__).resolve()
            if path.parent != PACKAGE_ROOT / "pipedream_solver":
                raise AssertionError(f"Module came from outside this bundle: {name}")
            imports.append({
                "module": name, "path": path.relative_to(PACKAGE_ROOT).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
            print(f"Imported pipedream_solver.{name}", flush=True)
        suite = unittest.defaultTestLoader.discover(str(PACKAGE_ROOT / "tests"))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    dependencies = {}
    for requirement in (PACKAGE_ROOT / "requirements.txt").read_text().splitlines():
        if not requirement or requirement.startswith("#"):
            continue
        name, expected = requirement.split("==")
        installed = importlib.metadata.version(name)
        if installed != expected:
            raise AssertionError(f"Dependency mismatch: {name}={installed}, expected={expected}")
        dependencies[name] = installed
    report = {
        "passed": result.wasSuccessful() and not result.skipped,
        "python": platform.python_version(), "system": platform.system(), "machine": platform.machine(),
        "module_count": len(imports), "modules": imports, "dependencies": dependencies,
        "tests_run": result.testsRun, "failures": len(result.failures),
        "errors": len(result.errors), "skipped": len(result.skipped),
        "scope": "Imports and representative synthetic smoke tests, not hydraulic calibration",
        "known_limitations": [
            "Transient regular and accelerated solvers are not assumed numerically interchangeable",
            "Water-quality smoke test uses rect_open sections and positive end lengths; not the WXQ full-line model",
            "Original unaccelerated Green-Ampt may fail Newton iteration for initially dry soil under heavy rain",
        ],
    }
    report_path = args.output / "library_validation.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "report": str(report_path)}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
