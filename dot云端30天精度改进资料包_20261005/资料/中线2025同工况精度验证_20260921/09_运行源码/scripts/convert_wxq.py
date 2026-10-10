from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from wxq_pipedream.wxq_to_superlink import parse_wxq_json


def load_model(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if "baseData" in raw:
        return raw
    if not raw:
        raise ValueError("WXQ JSON为空")
    _, model = next(iter(raw.items()))
    if not isinstance(model, dict) or "baseData" not in model:
        raise ValueError("无法识别WXQ JSON的baseData")
    return model


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert WXQ JSON to Pipedream input tables")
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--section-geometry-mode", choices=("safe", "legacy"), default="safe")
    parser.add_argument("--all-superjunction-depth", type=float)
    parser.add_argument("--minimum-initial-depth", type=float, default=0.01)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    model = load_model(args.model)
    gates = (model.get("baseData", {}) or {}).get("gates", {}) or {}
    gate_max_openings = {
        name: float(gate.get("b", 1.0) or 1.0)
        for name, gate in gates.items()
        if isinstance(gate, dict)
    }
    converted = parse_wxq_json(
        model,
        gate_max_openings=gate_max_openings,
        section_geometry_mode=args.section_geometry_mode,
        minimum_initial_depth_m=args.minimum_initial_depth,
    )
    if args.all_superjunction_depth is not None:
        depth = float(args.all_superjunction_depth)
        if not math.isfinite(depth) or depth <= 0:
            raise ValueError("--all-superjunction-depth必须是有限正数")
        converted["superjunctions"].loc[:, "h_0"] = depth
        converted["superjunctions"].loc[:, "max_depth"] = np.maximum(
            converted["superjunctions"]["max_depth"].to_numpy(dtype=float),
            depth,
        )

    for key, filename in (
        ("superjunctions", "superjunctions.csv"),
        ("superlinks", "superlinks.csv"),
        ("junctions", "junctions.csv"),
        ("links", "links.csv"),
        ("orifices", "orifices.csv"),
        ("pumps_df", "pumps.csv"),
    ):
        frame = converted.get(key)
        if frame is not None:
            frame.to_csv(args.output / filename, index=False, encoding="utf-8-sig")

    (args.output / "transects.json").write_text(
        json.dumps(json_safe(converted.get("transects", {})), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    audit_rows = []
    for row in converted.get("section_geometry_audit", []):
        errors = row.get("area_relative_errors", {}) or {}
        audit_rows.append({
            **{key: value for key, value in row.items() if key != "area_relative_errors"},
            "area_error_25pct": errors.get("0.25"),
            "area_error_50pct": errors.get("0.5"),
            "area_error_75pct": errors.get("0.75"),
            "area_error_100pct": errors.get("1.0"),
        })
    pd.DataFrame(audit_rows).to_csv(
        args.output / "section_geometry_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    superjunctions = converted["superjunctions"]
    summary = {
        "status": "COMPLETED",
        "model_path": str(args.model.resolve()),
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "section_geometry_mode": converted["section_geometry_mode"],
        "superjunctions": len(superjunctions),
        "superlinks": len(converted["superlinks"]),
        "internal_junctions": len(converted["junctions"]),
        "internal_links": len(converted["links"]),
        "orifices": 0 if converted.get("orifices") is None else len(converted["orifices"]),
        "transects": len(converted.get("transects", {})),
        "shape_counts": converted["links"]["shape"].value_counts().to_dict(),
        "geometry_audit_rows": len(converted.get("section_geometry_audit", [])),
        "geometry_fallbacks": len(converted.get("section_geometry_fallbacks", [])),
        "geometry_issues": len(converted.get("section_geometry_issues", [])),
        "superjunction_h0_min_m": float(superjunctions["h_0"].min()),
        "superjunction_h0_max_m": float(superjunctions["h_0"].max()),
    }
    (args.output / "conversion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
