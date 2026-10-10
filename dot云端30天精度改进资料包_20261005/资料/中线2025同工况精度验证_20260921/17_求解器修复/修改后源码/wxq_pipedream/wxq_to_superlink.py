"""Convert WXQ 1D hydraulic-model JSON into Pipedream SuperLink tables."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np  # noqa: F401
import pandas as pd

from .section_geometry import (
    SectionGeometry,
    classify_section,
    interpolate_section_geometry,
    raw_yz_points,
    to_pipedream_geometry,
)


DEFAULT_MINIMUM_INITIAL_DEPTH_M = 0.01
MIN_NUMERICAL_DEPTH_M = 0.01


def _initial_depth_from_level(
    initial_level: float | None,
    invert_elevation: float,
    *,
    minimum_depth_m: float = DEFAULT_MINIMUM_INITIAL_DEPTH_M,
) -> float:
    """Convert an absolute water level to a numerically usable cold-start depth."""
    if initial_level is None:
        return minimum_depth_m
    depth = float(initial_level) - float(invert_elevation)
    if not math.isfinite(depth) or depth <= 0.0:
        return minimum_depth_m
    return max(depth, minimum_depth_m)


def _gate_group_name(gate_name: str) -> str:
    match = re.match(r"JZZ\d+-(.+)-\d+$", gate_name)
    if match:
        return match.group(1)
    match = re.match(r"JZZ\d+-(.+?)(\d+)#$", gate_name)
    return match.group(1) if match else gate_name


def turbine_discharge_from_curve(curve_data: list, operating_value: float, head: float) -> float:
    """Interpolate turbine discharge from WXQ [operating value, head, flow] rows."""
    if float(operating_value) <= 0.0:
        return 0.0

    rows = np.asarray([row[:3] for row in (curve_data or []) if len(row) >= 3], dtype=float)
    if rows.size == 0:
        return 0.0
    rows = rows[np.all(np.isfinite(rows), axis=1)]
    if rows.size == 0:
        return 0.0

    operating_levels = np.unique(rows[:, 0])
    target_operating = float(np.clip(operating_value, operating_levels[0], operating_levels[-1]))
    lower = float(operating_levels[operating_levels <= target_operating][-1])
    upper = float(operating_levels[operating_levels >= target_operating][0])

    def flow_at(level: float) -> float:
        level_rows = rows[rows[:, 0] == level]
        order = np.argsort(level_rows[:, 1])
        heads = level_rows[order, 1]
        flows = level_rows[order, 2]
        return float(np.interp(float(head), heads, flows, left=flows[0], right=flows[-1]))

    lower_flow = flow_at(lower)
    if upper == lower:
        return max(lower_flow, 0.0)
    upper_flow = flow_at(upper)
    ratio = (target_operating - lower) / (upper - lower)
    return max(lower_flow + ratio * (upper_flow - lower_flow), 0.0)


def _trapezoid_from_yz(yz: list) -> dict[str, float] | None:
    if len(yz) < 3:
        return None
    pts = [(float(y), float(z)) for y, z in yz]
    y_vals = [p[0] for p in pts]
    z_vals = [p[1] for p in pts]
    min_z = min(z_vals)
    max_z = max(z_vals)
    height = max(max_z - min_z, 0.01)
    bottom_pts = [y for y, z in pts if abs(z - min_z) <= 1e-6]
    if len(bottom_pts) >= 2:
        bottom_width = max(bottom_pts) - min(bottom_pts)
    else:
        bottom_width = max(y_vals) - min(y_vals)
    top_width = max(y_vals) - min(y_vals)
    side_slope = max((top_width - bottom_width) / (2.0 * height), 0.0)
    return {
        "z_inv": min_z,
        "height": max(height, 0.01),
        "bottom_width": max(bottom_width, 0.01),
        "side_slope": side_slope,
    }


def _normalized_yz_points(yz: list) -> list[tuple[float, float]]:
    """Return finite WXQ cross-section points ordered by horizontal station."""
    points: dict[float, float] = {}
    for point in yz or []:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            station, elevation = float(point[0]), float(point[1])
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(station) and math.isfinite(elevation)):
            continue
        # A vertical bank can repeat a station. Pipedream's interpolator requires
        # unique x values, so retain the lower point as the hydraulic boundary.
        points[station] = min(points.get(station, elevation), elevation)
    return sorted(points.items())


def _transect_from_yz(
    yz: list,
    *,
    horiz_points: int | None = None,
    vert_points: int = 100,
) -> dict[str, list[float] | int] | None:
    """Convert absolute WXQ yz coordinates into a Pipedream irregular transect."""
    points = _normalized_yz_points(yz)
    if len(points) < 2 or points[-1][0] <= points[0][0]:
        return None

    min_station = points[0][0]
    min_elevation = min(elevation for _, elevation in points)
    if max(elevation for _, elevation in points) <= min_elevation:
        return None
    sample_count = horiz_points if horiz_points is not None else max(len(points), 100)
    return {
        "x": [station - min_station for station, _ in points],
        "y": [elevation - min_elevation for _, elevation in points],
        "horiz_points": min(max(int(sample_count), 2), 1000),
        "vert_points": min(max(int(vert_points), 2), 1000),
    }


def _interpolate_yz(start_yz: list, end_yz: list, fraction: float) -> list[list[float]]:
    """Interpolate two differently sampled cross-sections on normalized width."""
    start = _normalized_yz_points(start_yz)
    end = _normalized_yz_points(end_yz)
    if len(start) < 2 or len(end) < 2:
        raise ValueError("不规则断面插值至少需要两个横向坐标不同的 yz 点")

    fraction = min(max(float(fraction), 0.0), 1.0)

    def normalized_profile(points: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray, float]:
        stations = np.asarray([point[0] for point in points], dtype=float)
        elevations = np.asarray([point[1] for point in points], dtype=float)
        width = float(stations[-1] - stations[0])
        if width <= 0.0:
            raise ValueError("不规则断面宽度必须大于 0")
        return (stations - stations[0]) / width, elevations, width

    start_stations, start_elevations, start_width = normalized_profile(start)
    end_stations, end_elevations, end_width = normalized_profile(end)
    stations = np.unique(np.concatenate((start_stations, end_stations)))
    if len(stations) > 1000:
        stations = np.linspace(0.0, 1.0, 1000)

    width = start_width + (end_width - start_width) * fraction
    start_z = np.interp(stations, start_stations, start_elevations)
    end_z = np.interp(stations, end_stations, end_elevations)
    elevations = start_z + (end_z - start_z) * fraction
    return [[float(station * width), float(elevation)] for station, elevation in zip(stations, elevations)]


def _interpolate_geometry(start: dict[str, float], end: dict[str, float], fraction: float) -> dict[str, float]:
    fraction = min(max(float(fraction), 0.0), 1.0)
    return {
        key: float(start[key]) + (float(end[key]) - float(start[key])) * fraction
        for key in ("z_inv", "height", "bottom_width", "side_slope")
    }


def _channel_key_from_node(node_name: str) -> str | None:
    match = re.search(r"(CH\d+[^（(]*)", node_name)
    return match.group(1).strip() if match else None


def _full_trapezoid_area(height: float, bottom_width: float, side_slope: float) -> float:
    return height * (bottom_width + side_slope * height)


def _full_trapezoid_perimeter(height: float, bottom_width: float, side_slope: float) -> float:
    return bottom_width + 2.0 * height * math.sqrt(1.0 + side_slope**2)


def _equivalent_length_for_local_loss(length: float, loss_coeff: float, nc: float, height: float, bottom_width: float, side_slope: float) -> float:
    area = _full_trapezoid_area(height, bottom_width, side_slope)
    perimeter = _full_trapezoid_perimeter(height, bottom_width, side_slope)
    if area <= 0 or perimeter <= 0 or loss_coeff <= 0:
        return length
    hydraulic_radius = area / perimeter
    hydraulic_diameter = 4.0 * hydraulic_radius
    # Manning n 粗略折成 Darcy-Weisbach f，使 K·V²/2g = f·L/D·V²/2g。
    darcy_f = max(8.0 * 9.81 * (nc**2) / max(hydraulic_radius, 1e-6) ** (1.0 / 3.0), 1e-4)
    return length + loss_coeff * hydraulic_diameter / darcy_f


def _siphon_geometry(siphon: dict) -> tuple[str, float, float, float, float, float, int]:
    """Map WXQ siphon barrel geometry to Pipedream geometry fields.

    Returns shape, g1, g2, g3, single-barrel width, single-barrel height, barrel count.
    For closed rectangular barrels, g3 is the Preissmann slot width ratio.
    """
    slot_width = float(siphon.get("preissmann_slot", siphon.get("slot_width", 0.001)) or 0.001)
    shape_hint = str(
        siphon.get("shape")
        or siphon.get("geometry")
        or siphon.get("section_shape")
        or ""
    ).strip().lower().replace("-", "_").replace(" ", "_")
    shape_aliases = {
        "box": "rect_closed",
        "box_culvert": "rect_closed",
        "rect": "rect_closed",
        "rectangle": "rect_closed",
        "rectangular": "rect_closed",
        "rectangular_closed": "rect_closed",
        "closed_rect": "rect_closed",
        "closed_rectangle": "rect_closed",
        "circle": "force_main",
        "round": "force_main",
        "pipe": "force_main",
        "pressure_pipe": "force_main",
        "force_main": "force_main",
        "circular": "force_main",
        "trapezoid": "trapezoidal",
        "trapezoidal": "trapezoidal",
    }
    shape_hint = shape_aliases.get(shape_hint, shape_hint)

    diameter = None
    for key in ("diameter", "d_orifice", "D", "d", "pipe_diameter"):
        if siphon.get(key) is not None:
            diameter = float(siphon.get(key) or 0)
            break

    n_orifice = max(int(float(siphon.get("n_orifice", 1) or 1)), 1)
    b_ori = float(siphon.get("b_orifice", siphon.get("inlet_b", diameter or 2.0)) or 2.0)
    h_ori = float(siphon.get("h_orifice", diameter or b_ori) or b_ori)
    total_width = max(b_ori * n_orifice, 0.01)
    height = max(h_ori, 0.01)
    side_slope = max((float(siphon.get("inlet_m", 0.0) or 0.0) + float(siphon.get("outlet_m", 0.0) or 0.0)) / 2.0, 0.0)

    if shape_hint in {"force_main", "circular"} or diameter:
        pipe_diameter = max(float(diameter or h_ori or b_ori), 0.01)
        return "force_main", pipe_diameter, slot_width, 0.0, b_ori, h_ori, n_orifice

    if shape_hint == "trapezoidal":
        return "trapezoidal", height, total_width, side_slope, b_ori, h_ori, n_orifice

    # WXQ siphons with b_orifice/h_orifice/n_orifice describe closed box barrels.
    # Pipedream's rect_closed shape matches that physical meaning better than
    # the open-channel trapezoidal approximation.
    if shape_hint in {"", "rect_closed"} and b_ori > 0 and h_ori > 0:
        return "rect_closed", height, total_width, slot_width, b_ori, h_ori, n_orifice

    return "trapezoidal", height, total_width, side_slope, b_ori, h_ori, n_orifice


def parse_wxq_json(
    json_path: Path | str | dict,
    gate_max_openings: dict[str, float] | None = None,
    section_geometry_mode: str = "safe",
    minimum_initial_depth_m: float = DEFAULT_MINIMUM_INITIAL_DEPTH_M,
) -> dict:
    """解析王孝群 1D 模型 JSON，返回 Pipedream 兼容拓扑 dict。

    返回 dict 包含：
      superjunctions: pd.DataFrame — SuperLink 节点表
      superlinks: pd.DataFrame — SuperLink 管道表
      orifices: pd.DataFrame | None — 孔口表（闸门 + 阀门）
      model_name: str
      timeControl: dict
      bleeders: list[dict] — 分水口元数据
      pumps: list[dict] — 泵站元数据
      siphons_info: list[dict] — 倒虹吸原始参数
      boundaries: dict — 边界条件
      initial_data: dict — 初始水位/流量
      time_series: dict — 时变边界序列
    """
    if section_geometry_mode not in {"safe", "legacy"}:
        raise ValueError("section_geometry_mode 必须是 safe 或 legacy")
    if isinstance(minimum_initial_depth_m, bool) or not math.isfinite(minimum_initial_depth_m) or minimum_initial_depth_m <= 0:
        raise ValueError("minimum_initial_depth_m must be finite and positive")
    if isinstance(json_path, dict):
        raw = json_path
        source_name = "wxq_model"
    else:
        source_path = Path(json_path)
        raw = json.loads(source_path.read_text(encoding="utf-8"))
        source_name = source_path.stem
    if isinstance(raw, dict) and "baseData" in raw:
        model_name, model_obj = source_name, raw
    else:
        model_name, model_obj = next(iter(raw.items()))
    bd = model_obj.get("baseData", {})
    raw_nodes = bd.get("nodes", {}) or {}
    nodes = {
        name: dict(node) if isinstance(node, dict) else {}
        for name, node in raw_nodes.items()
    }
    channels = bd.get("channels", {}) or {}
    gates = bd.get("gates", {}) or {}
    siphons = bd.get("siphons", {}) or {}

    # 个别历史 WXQ 模型把连接节点底高程留成 0。用相邻构筑物的绝对
    # 底高程修复，避免闸门相对高程和初始水深被放大几十米。
    topology_repairs: list[dict] = []
    for node_name, node in nodes.items():
        original_z = float(node.get("zb", 0.0) or 0.0)
        if original_z > 0:
            continue
        candidates: list[float] = []
        for gate in gates.values():
            if isinstance(gate, dict) and node_name in {gate.get("node1"), gate.get("node2")}:
                candidate = float(gate.get("zb", 0.0) or 0.0)
                if candidate > 0:
                    candidates.append(candidate)
        for siphon in siphons.values():
            if not isinstance(siphon, dict):
                continue
            if siphon.get("node1") == node_name:
                candidate = float(siphon.get("inlet_zb", 0.0) or 0.0)
                if candidate > 0:
                    candidates.append(candidate)
            if siphon.get("node2") == node_name:
                candidate = float(siphon.get("outlet_zb", 0.0) or 0.0)
                if candidate > 0:
                    candidates.append(candidate)
        if candidates:
            repaired_z = min(candidates)
            node["zb"] = repaired_z
            topology_repairs.append({
                "object_type": "node",
                "object_name": node_name,
                "field": "zb",
                "original": original_z,
                "repaired": repaired_z,
                "reason": "使用相邻闸门或倒虹吸底高程修复零值",
            })
    raw_sections = bd.get("sections", {}) or {}
    sections: dict[str, dict] = {}
    for section_key, section_value in raw_sections.items():
        if isinstance(section_value, dict):
            sections[section_key] = section_value
            continue
        if not isinstance(section_value, list):
            continue

        # 兼容旧版 WXQ：sections[channel_name] = [section, ...]。
        generated_names = []
        first_x = first_y = None
        for index, item in enumerate(section_value):
            if not isinstance(item, dict):
                continue
            generated_name = f"{section_key}__section_{index}"
            normalized = dict(item)
            x = float(item.get("x1", item.get("x", 0.0)) or 0.0)
            y = float(item.get("y1", item.get("y", 0.0)) or 0.0)
            if first_x is None:
                first_x, first_y = x, y
            normalized.setdefault("location", math.hypot(x - first_x, y - first_y))
            normalized.setdefault("num_interp", 0)
            sections[generated_name] = normalized
            generated_names.append(generated_name)
        if section_key in channels and not channels[section_key].get("sec_names"):
            channels[section_key]["sec_names"] = generated_names
    init_data = model_obj.get("initialData", {})
    init_channels = init_data.get("channels", {})
    local_z = init_channels.get("local_Z", {}) or {}
    gate_max_openings = gate_max_openings or {}

    # local_Z 的 key 是渠道名，value 是该渠道 node1 的控制水位。
    # Pipedream max_depth 使用相对节点底高程的水深，不能直接使用绝对水位。
    node_max_depth: dict[str, float] = {}
    for channel_name, local_water_level in local_z.items():
        channel = channels.get(channel_name)
        if not isinstance(channel, dict):
            continue
        node_name = channel.get("node1")
        node = nodes.get(node_name)
        if not isinstance(node, dict):
            continue
        water_level = float(local_water_level)
        node_zb = float(node.get("zb", 0.0) or 0.0)
        derived_depth = water_level - node_zb
        if derived_depth > 0.0:
            node_max_depth[node_name] = max(node_max_depth.get(node_name, 0.0), derived_depth)
            continue
        topology_repairs.append({
            "object_type": "node",
            "object_name": node_name,
            "field": "max_depth",
            "original": derived_depth,
            "repaired": None,
            "reason": (
                f"渠道 {channel_name} 的 local_Z={water_level} 不高于节点底高程 "
                f"zb={node_zb}，未使用该非正水深限制 Pipedream"
            ),
        })

    # 节点映射
    name2id = {name: i for i, name in enumerate(nodes.keys())}

    # --- 从 sections 提取梯形断面几何：g1=高度, g2=底宽, g3=边坡(run/rise) ---
    section_geom = {}
    section_yz: dict[str, list[list[float]]] = {}
    section_geometries: dict[str, SectionGeometry] = {}
    section_geometry_issues: list[dict] = []
    for sec_name, sec in sections.items():
        if section_geometry_mode == "safe":
            geometry = classify_section(sec.get("yz", []), source_name=sec_name)
            if geometry is None:
                section_geometry_issues.append({
                    "section": sec_name,
                    "reason": "断面无有效宽度、高度或存在自交",
                })
                continue
            section_geometries[sec_name] = geometry
            section_geom[sec_name] = geometry.analytic_geometry()
            section_yz[sec_name] = [list(point) for point in geometry.raw_yz]
        else:
            geom = _trapezoid_from_yz(sec.get("yz", []))
            if geom:
                section_geom[sec_name] = geom
            yz_points = _normalized_yz_points(sec.get("yz", []))
            if len(yz_points) >= 2 and yz_points[-1][0] > yz_points[0][0]:
                section_yz[sec_name] = [[station, elevation] for station, elevation in yz_points]

    def _channel_geometry(ch: dict) -> tuple[str, float, float, float]:
        """返回渠道级代表几何，仅用于 SuperLink 元数据和回退路径。"""
        sec_names = ch.get("sec_names", [])
        heights, bottom_widths, side_slopes = [], [], []
        for sn in sec_names:
            if sn in section_geom:
                heights.append(section_geom[sn]["height"])
                bottom_widths.append(section_geom[sn]["bottom_width"])
                side_slopes.append(section_geom[sn]["side_slope"])
        if heights:
            avg_h = sum(heights) / len(heights)
            avg_b = sum(bottom_widths) / len(bottom_widths)
            avg_m = sum(side_slopes) / len(side_slopes)
            return "trapezoidal", avg_h, avg_b, avg_m
        return "trapezoidal", 1.0, 1.0, 0.0

    def _channel_section_profile(ch: dict) -> list[dict] | None:
        """按 WXQ num_interp 规则构造沿程计算断面。

        num_interp 定义在区间上游断面，值为 n 时，在该断面与下一
        实测断面之间按里程比例插入 n 个计算断面。
        """
        sec_names = [
            name
            for name in ch.get("sec_names", [])
            if name in section_geom and name in section_yz and name in sections
        ]
        if len(sec_names) < 2:
            return None

        source_points = []
        for sec_name in sec_names:
            section = sections[sec_name]
            location = section.get("location")
            if location is None:
                return None
            source_points.append(
                {
                    "name": sec_name,
                    "location": float(location),
                    "geometry": section_geom[sec_name],
                    "yz": section_yz[sec_name],
                    "section_geometry": section_geometries.get(sec_name),
                    "num_interp": max(int(section.get("num_interp", 0) or 0), 0),
                }
            )

        profile: list[dict] = []
        cumulative_distance = 0.0
        for index in range(len(source_points) - 1):
            start = source_points[index]
            end = source_points[index + 1]
            interval_length = abs(end["location"] - start["location"])
            if interval_length <= 0:
                raise ValueError(f"渠道相邻断面里程必须不同: {start['name']} -> {end['name']}")

            subdivisions = start["num_interp"] + 1
            for subdivision in range(subdivisions):
                fraction = subdivision / subdivisions
                if section_geometry_mode == "safe":
                    section_geometry = interpolate_section_geometry(
                        start["section_geometry"], end["section_geometry"], fraction
                    )
                    geometry = section_geometry.analytic_geometry()
                    yz = [list(point) for point in section_geometry.raw_yz]
                else:
                    section_geometry = None
                    geometry = _interpolate_geometry(start["geometry"], end["geometry"], fraction)
                    yz = _interpolate_yz(start["yz"], end["yz"], fraction)
                profile.append(
                    {
                        "name": start["name"] if fraction == 0 else None,
                        "source_interval": [start["name"], end["name"]],
                        "fraction": fraction,
                        "distance": cumulative_distance + interval_length * fraction,
                        "geometry": geometry,
                        "yz": yz,
                        "section_geometry": section_geometry,
                        "interpolated": fraction > 0,
                    }
                )
            cumulative_distance += interval_length

        last = source_points[-1]
        profile.append(
            {
                "name": last["name"],
                "source_interval": [last["name"], last["name"]],
                "fraction": 1.0,
                "distance": cumulative_distance,
                "geometry": last["geometry"],
                "yz": last["yz"],
                "section_geometry": last["section_geometry"],
                "interpolated": False,
            }
        )
        return profile

    node_local_z: dict[str, float] = {}
    for ch_name, ch in channels.items():
        z = local_z.get(ch_name)
        if z is None:
            continue
        for node_name in (ch.get("node1"), ch.get("node2")):
            if node_name and node_name not in node_local_z:
                node_local_z[node_name] = float(z)
    for node_name in nodes:
        if node_name in node_local_z:
            continue
        key = _channel_key_from_node(node_name)
        if key:
            for ch_name, z in local_z.items():
                if ch_name.startswith(key):
                    node_local_z[node_name] = float(z)
                    break

    # 用已知节点水位沿拓扑传播，减少没有 local_Z 的中间节点冷启动回退。
    # 只有 nodeType=1 的水位边界可以覆盖初始水位；
    # nodeType=2 的 boundary 值是流量，不能混入水位。
    boundaries = init_data.get("boundaries", {}) or {}
    for node_name, value in boundaries.items():
        node = nodes.get(node_name)
        if isinstance(node, dict) and int(node.get("nodeType", 0) or 0) == 1:
            node_local_z[node_name] = float(value)
    adjacency: dict[str, set[str]] = {name: set() for name in nodes}
    for collection in (channels, gates, siphons):
        for item in collection.values():
            if not isinstance(item, dict):
                continue
            n1, n2 = item.get("node1"), item.get("node2")
            if n1 in adjacency and n2 in adjacency:
                adjacency[n1].add(n2)
                adjacency[n2].add(n1)
    for _ in range(len(nodes)):
        inferred = {}
        for node_name, neighbours in adjacency.items():
            if node_name in node_local_z:
                continue
            known = [node_local_z[name] for name in neighbours if name in node_local_z]
            if known:
                inferred[node_name] = sum(known) / len(known)
        if not inferred:
            break
        node_local_z.update(inferred)

    # --- superjunctions ---
    sj_rows = []
    for i, (name, n) in enumerate(nodes.items()):
        zb = float(n.get("zb", 0))
        amin = max(float(n.get("Amin", 10)), 10.0)
        nt = int(n.get("nodeType", 0))
        initial_z = node_local_z.get(name)
        minimum_depth = MIN_NUMERICAL_DEPTH_M if nt == 1 else minimum_initial_depth_m
        h_0 = _initial_depth_from_level(initial_z, zb, minimum_depth_m=minimum_depth)
        configured_max_depth = n.get("max_depth", n.get("maxDepth"))
        if configured_max_depth is not None:
            max_depth = float(configured_max_depth)
        else:
            max_depth = node_max_depth.get(name, np.inf)
        max_depth = max(max_depth, h_0)
        sj_rows.append({
            "name": name, "id": i, "z_inv": zb, "h_0": h_0,
            # WXQ nodeType=1 为给定水位，nodeType=2 为给定流量。
            "bc": nt == 1, "storage": "functional",
            "a": 0.0, "b": 0.0, "c": amin, "max_depth": max_depth,
            "A_s": amin,
        })

    # --- 蛟流河顶管修正：报告中是倒虹吸（有压隧洞），JSON 误标为 channel ---
    SIPHON_CHANNEL_KEYWORDS = ("蛟流河", "顶管")

    def _is_siphon_channel(ch_name: str) -> bool:
        return any(kw in ch_name for kw in SIPHON_CHANNEL_KEYWORDS)

    # --- 用 sections 里程差计算渠道真实长度 ---
    def _channel_length_from_sections(ch: dict) -> float:
        """从渠道关联的断面里程推算真实长度（比坐标直线更准确）。"""
        sec_names = ch.get("sec_names", [])
        locs = []
        for sn in sec_names:
            if sn in sections:
                loc = sections[sn].get("location", None)
                if loc is not None:
                    locs.append(float(loc))
        if len(locs) >= 2:
            return abs(max(locs) - min(locs))
        return 0.0

    # --- superlinks (channels) ---
    sl_rows = []
    internal_profiles: dict[int, dict] = {}
    link_id = 0
    siphon_from_channel = []  # 从 channel 转为 siphon 的记录

    for name, ch in channels.items():
        n1, n2 = ch.get("node1", ""), ch.get("node2", "")
        if n1 not in name2id or n2 not in name2id:
            continue
        nc = float(ch.get("nc", 0.013))

        # 优先按 WXQ 断面和 num_interp 构造沿程计算断面。
        section_profile = _channel_section_profile(ch)
        dx = section_profile[-1]["distance"] if section_profile else _channel_length_from_sections(ch)
        if dx < 10.0:
            nd1, nd2 = nodes[n1], nodes[n2]
            dx = _coordinate_distance(float(nd1.get("x", 0)), float(nd1.get("y", 0)),
                                      float(nd2.get("x", 0)), float(nd2.get("y", 0)))
        dx = max(dx, 10.0)
        if section_profile and section_profile[-1]["distance"] != dx:
            original_length = max(float(section_profile[-1]["distance"]), 1e-9)
            for profile_point in section_profile:
                profile_point["distance"] = float(profile_point["distance"]) * dx / original_length

        # 蛟流河顶管 → 转为有压管道（circular）
        shape, g1, g2, g3 = _channel_geometry(ch)
        h0_link = max((sj_rows[name2id[n1]]["h_0"] + sj_rows[name2id[n2]]["h_0"]) / 2.0, 0.01)
        if not section_profile:
            section_profile = [
                {
                    "name": None,
                    "source_interval": [None, None],
                    "fraction": 0.0,
                    "distance": 0.0,
                    "geometry": {
                        "z_inv": float(nodes[n1].get("zb", 0.0)),
                        "height": g1,
                        "bottom_width": g2,
                        "side_slope": g3,
                    },
                    "interpolated": False,
                },
                {
                    "name": None,
                    "source_interval": [None, None],
                    "fraction": 1.0,
                    "distance": dx,
                    "geometry": {
                        "z_inv": float(nodes[n2].get("zb", 0.0)),
                        "height": g1,
                        "bottom_width": g2,
                        "side_slope": g3,
                    },
                    "interpolated": False,
                },
            ]
        if _is_siphon_channel(name):
            nc = min(nc, 0.015)  # 有压管道糙率不应超过 0.015
            sl_rows.append({
                "name": name, "id": link_id,
                "sj_0": name2id[n1], "sj_1": name2id[n2],
                "in_offset": 0.0, "out_offset": 0.0,
                "dx": dx, "n": nc,
                "shape": shape, "g1": g1, "g2": g2,
                "g3": g3, "g4": 0.0,
                "Q_0": 0.0, "h_0": h0_link, "ctrl": False,
                "A_s": 0.0, "A_c": 0.0, "C": 1.0,
            })
            siphon_from_channel.append({
                "name": name, "node1": n1, "node2": n2,
                "length": dx, "shape": shape, "height": g1, "bottom_width": g2, "side_slope": g3, "nc": nc,
                "reason": "channel 名含倒虹吸/顶管关键词，按梯形有压通道近似处理",
            })
        else:
            sl_rows.append({
                "name": name, "id": link_id,
                "sj_0": name2id[n1], "sj_1": name2id[n2],
                "in_offset": 0.0, "out_offset": 0.0,
                "dx": dx, "n": nc,
                "shape": shape, "g1": g1, "g2": g2,
                "g3": g3, "g4": 0.0,
                "Q_0": 0.0, "h_0": h0_link, "ctrl": False,
                "A_s": 0.0, "A_c": 0.0, "C": 1.0,
            })
        internal_profiles[link_id] = {
            "name": name,
            "profile": section_profile,
            "roughness": nc,
            "shape": shape,
            "Q_0": 0.0,
            "upstream_head": float(nodes[n1].get("zb", 0.0)) + sj_rows[name2id[n1]]["h_0"],
            "downstream_head": float(nodes[n2].get("zb", 0.0)) + sj_rows[name2id[n2]]["h_0"],
        }
        link_id += 1

    # --- superlinks (siphons → 有压管道) ---
    siphon_info = []
    for name, s in siphons.items():
        if not isinstance(s, dict):
            continue
        n1 = s.get("node1", "")
        n2 = s.get("node2", "")
        if n1 not in name2id or n2 not in name2id:
            continue

        length = float(s.get("l_orifice", 0))
        if length <= 0:
            nd1, nd2 = nodes[n1], nodes[n2]
            length = _coordinate_distance(float(nd1.get("x", 0)), float(nd1.get("y", 0)),
                                          float(nd2.get("x", 0)), float(nd2.get("y", 0)))
        length = max(length, 10.0)

        nc = float(s.get("nc", 0.015))
        shape, g1, g2, g3, b_ori, h_ori, n_ori = _siphon_geometry(s)
        loss_coeff = float(s.get("loss_coeff", 0.0) or 0.0)
        physical_length = length
        equivalent_side_slope = g3 if shape == "trapezoidal" else 0.0
        equivalent_width = g2 if shape != "force_main" else g1
        length = _equivalent_length_for_local_loss(length, loss_coeff, nc, g1, equivalent_width, equivalent_side_slope)
        h0_link = max((sj_rows[name2id[n1]]["h_0"] + sj_rows[name2id[n2]]["h_0"]) / 2.0, 0.01)

        sl_rows.append({
            "name": f"siphon_{name}", "id": link_id,
            "sj_0": name2id[n1], "sj_1": name2id[n2],
            "in_offset": 0.0, "out_offset": 0.0,
            "dx": length, "n": nc,
            "shape": shape, "g1": g1, "g2": g2,
            "g3": g3, "g4": 0.0,
            "Q_0": 0.0, "h_0": h0_link, "ctrl": False,
            "A_s": 0.0, "A_c": 0.0, "C": 1.0,
        })
        internal_profiles[link_id] = {
            "name": f"siphon_{name}",
            "profile": [
                {
                    "name": None,
                    "source_interval": [name, name],
                    "fraction": 0.0,
                    "distance": 0.0,
                    "geometry": {
                        "z_inv": float(s.get("inlet_zb", nodes[n1].get("zb", 0.0)) or nodes[n1].get("zb", 0.0)),
                        "height": g1,
                        "bottom_width": g2,
                        "side_slope": g3,
                    },
                    "interpolated": False,
                },
                {
                    "name": None,
                    "source_interval": [name, name],
                    "fraction": 1.0,
                    "distance": length,
                    "geometry": {
                        "z_inv": float(s.get("outlet_zb", nodes[n2].get("zb", 0.0)) or nodes[n2].get("zb", 0.0)),
                        "height": g1,
                        "bottom_width": g2,
                        "side_slope": g3,
                    },
                    "interpolated": False,
                },
            ],
            "roughness": nc,
            "shape": shape,
            "Q_0": 0.0,
            "upstream_head": float(nodes[n1].get("zb", 0.0)) + sj_rows[name2id[n1]]["h_0"],
            "downstream_head": float(nodes[n2].get("zb", 0.0)) + sj_rows[name2id[n2]]["h_0"],
        }
        link_id += 1

        siphon_info.append({
            "name": name,
            "node1": n1, "node2": n2,
            "length": length,
            "physical_length": physical_length,
            "equivalent_length_added": length - physical_length,
            "shape": shape, "g1": g1, "g2": g2, "g3": g3,
            "width": b_ori, "height": h_ori, "n_orifice": n_ori,
            "nc": nc,
            "loss_coeff": loss_coeff,
            "inlet_zb": float(s.get("inlet_zb", 0)),
            "outlet_zb": float(s.get("outlet_zb", 0)),
            "sj_0": name2id[n1], "sj_1": name2id[n2],
            "superlink_id": link_id - 1,
        })

    # --- explicit internals: mirror WXQ section discretization ---
    internal_link_rows: list[dict] = []
    internal_junction_rows: list[dict] = []
    channel_discretization: dict[str, dict] = {}
    transects: dict[str, dict] = {}
    section_geometry_audit: list[dict] = []
    superlink_flow_signs: dict[int, float] = {}
    next_junction_id = 0
    next_internal_link_id = 0

    for superlink_row in sl_rows:
        superlink_id = int(superlink_row["id"])
        spec = internal_profiles[superlink_id]
        profile = [dict(point) for point in spec["profile"]]
        total_length = max(float(profile[-1]["distance"]), 1e-9)

        # Pipedream 会把底高程较高的一端作为名义上游；显式内部表也必须同步反向。
        # External WXQ outputs retain the original node1 -> node2 flow direction.
        reversed_for_solver = float(profile[0]["geometry"]["z_inv"]) < float(profile[-1]["geometry"]["z_inv"])
        flow_sign = -1.0 if reversed_for_solver else 1.0
        superlink_flow_signs[superlink_id] = flow_sign
        if reversed_for_solver:
            profile = list(reversed(profile))
            for point in profile:
                point["distance"] = total_length - float(point["distance"])
            profile.sort(key=lambda point: point["distance"])
            superlink_row["sj_0"], superlink_row["sj_1"] = superlink_row["sj_1"], superlink_row["sj_0"]
            spec["upstream_head"], spec["downstream_head"] = spec["downstream_head"], spec["upstream_head"]

        upstream_node_z = float(sj_rows[int(superlink_row["sj_0"])]["z_inv"])
        downstream_node_z = float(sj_rows[int(superlink_row["sj_1"])]["z_inv"])
        superlink_row["in_offset"] = float(profile[0]["geometry"]["z_inv"]) - upstream_node_z
        superlink_row["out_offset"] = float(profile[-1]["geometry"]["z_inv"]) - downstream_node_z

        junction_ids = list(range(next_junction_id, next_junction_id + len(profile)))
        next_junction_id += len(profile)
        superlink_row["j_0"] = junction_ids[0]
        superlink_row["j_1"] = junction_ids[-1]

        for point, junction_id in zip(profile, junction_ids):
            position_fraction = float(point["distance"]) / total_length
            water_head = float(spec["upstream_head"]) + (
                float(spec["downstream_head"]) - float(spec["upstream_head"])
            ) * position_fraction
            z_inv = float(point["geometry"]["z_inv"])
            internal_junction_rows.append(
                {
                    "id": junction_id,
                    "k": superlink_id,
                    "h_0": _initial_depth_from_level(water_head, z_inv, minimum_depth_m=minimum_initial_depth_m),
                    "A_s": 0.0,
                    "z_inv": z_inv,
                }
            )

        first_internal_link_id = next_internal_link_id
        for interval_index in range(len(profile) - 1):
            start = profile[interval_index]
            end = profile[interval_index + 1]
            interval_length = float(end["distance"]) - float(start["distance"])
            if interval_length <= 0:
                raise ValueError(f"SuperLink 内部断面距离必须递增: {spec['name']}")
            transect_name = ""
            if (
                section_geometry_mode == "safe"
                and start.get("section_geometry") is not None
                and end.get("section_geometry") is not None
            ):
                midpoint_section = interpolate_section_geometry(
                    start["section_geometry"], end["section_geometry"], 0.5
                )
                serialized = to_pipedream_geometry(midpoint_section)
                midpoint_geometry = {
                    "z_inv": midpoint_section.z_inv,
                    "height": float(serialized["g1"]),
                    "bottom_width": float(serialized["g2"]),
                    "side_slope": float(serialized["g3"]),
                }
                link_shape = str(serialized["shape"])
                transect = serialized["transect"]
                if transect is not None:
                    transect_name = f"{spec['name']}__link_{next_internal_link_id}"
                    transects[transect_name] = transect
                section_geometry_audit.append({
                    "internal_link_id": next_internal_link_id,
                    "superlink_id": superlink_id,
                    "channel": spec["name"],
                    "source_interval": start.get("source_interval"),
                    "start_shape": start["section_geometry"].shape_kind,
                    "end_shape": end["section_geometry"].shape_kind,
                    "midpoint_shape": midpoint_section.shape_kind,
                    "output_shape": link_shape,
                    "target_height": midpoint_section.height,
                    "output_height": float(serialized["g1"]),
                    **serialized["quality"],
                })
            else:
                midpoint_geometry = _interpolate_geometry(start["geometry"], end["geometry"], 0.5)
                link_shape = spec["shape"]
                if start.get("yz") and end.get("yz"):
                    midpoint_yz = _interpolate_yz(start["yz"], end["yz"], 0.5)
                    transect = _transect_from_yz(midpoint_yz)
                    if transect is not None:
                        transect_name = f"{spec['name']}__link_{next_internal_link_id}"
                        transects[transect_name] = transect
                        link_shape = "irregular"
            internal_link_rows.append(
                {
                    "id": next_internal_link_id,
                    "j_0": junction_ids[interval_index],
                    "j_1": junction_ids[interval_index + 1],
                    "k": superlink_id,
                    "dx": interval_length,
                    "n": float(spec["roughness"]),
                    "shape": link_shape,
                    "transect": transect_name,
                    "g1": midpoint_geometry["height"],
                    "g2": midpoint_geometry["bottom_width"],
                    "g3": midpoint_geometry["side_slope"],
                    "g4": 0.0,
                    "Q_0": float(spec["Q_0"]),
                    "ctrl": False,
                    "A_c": 0.0,
                    "C": 1.0,
                    "friction_method": "cm",
                }
            )
            next_internal_link_id += 1

        channel_discretization[spec["name"]] = {
            "superlink_id": superlink_id,
            "source_section_count": sum(1 for point in profile if not point["interpolated"]),
            "interpolated_section_count": sum(1 for point in profile if point["interpolated"]),
            "calculation_section_count": len(profile),
            "internal_link_count": len(profile) - 1,
            "internal_link_ids": list(range(first_internal_link_id, next_internal_link_id)),
            "flow_sign": flow_sign,
            "sections": [
                {
                    "name": point["name"],
                    "junction_id": junction_id,
                    "flow_link_id": (
                        first_internal_link_id + min(index, max(len(profile) - 2, 0))
                    ),
                    "flow_sign": flow_sign,
                    "distance": float(point["distance"]),
                    "interpolated": bool(point["interpolated"]),
                    "source_interval": point["source_interval"],
                    "fraction": float(point["fraction"]),
                    "shape_kind": (
                        point["section_geometry"].shape_kind
                        if point.get("section_geometry") is not None
                        else None
                    ),
                    **{key: float(value) for key, value in point["geometry"].items()},
                }
                for index, (point, junction_id) in enumerate(zip(profile, junction_ids))
            ],
        }

    for siphon in siphon_info:
        siphon["flow_sign"] = superlink_flow_signs.get(int(siphon["superlink_id"]), 1.0)

    links_df = pd.DataFrame(internal_link_rows).set_index("id", drop=False)
    junctions_df = pd.DataFrame(internal_junction_rows).set_index("id", drop=False)

    # --- orifices (gates + valves) ---
    ori_rows = []
    ori_id = 0

    for name, g in gates.items():
        if not isinstance(g, dict):
            continue
        n1 = g.get("node1") or g.get("upNode", "")
        n2 = g.get("node2") or g.get("downNode", "")
        if n1 not in name2id or n2 not in name2id:
            continue
        b = float(g.get("b", 2.0))
        is_maintenance = "检修" in str(name)
        cd = float(g.get("c1", g.get("Cd", 0.95 if is_maintenance else 0.7)))
        group_name = _gate_group_name(name)
        y_max = float(gate_max_openings.get(group_name, gate_max_openings.get(name, b)) or b)
        gate_area = b * y_max
        # z_o 是相对于上游节点底高程(z_inv)的偏移，不是绝对高程！
        # 闸底通常与渠底齐平 → z_o = 0
        # 如果 JSON 给了 zb（绝对闸底高程），转为相对值
        zb_abs = float(g.get("zb", 0))
        z_inv_n1 = float(nodes[n1].get("zb", 0))
        z_o_relative = max(zb_abs - z_inv_n1, 0.0) if zb_abs > 0 else 0.0
        ori_rows.append({
            "id": ori_id, "name": name,
            "sj_0": name2id[n1], "sj_1": name2id[n2],
            "A": gate_area, "orientation": "side",
            "z_o": z_o_relative, "y_max": y_max,
            "shape": "rect_open", "g1": y_max, "g2": b, "g3": 0.0,
            "flow_model": "rectangular_pressure",
            "C": cd,
            "is_maintenance": is_maintenance,
            "default_opening": 1.0 if is_maintenance else None,
            "gate_type": g.get("type"),
            "c1": float(g.get("c1", cd)),
            "c2": float(g.get("c2", cd)),
            "c3": float(g.get("c3", cd)),
            "c4": float(g.get("c4", cd)),
            "group_name": group_name,
        })
        ori_id += 1

    valves = bd.get("valves", {}) or {}
    for name, v in valves.items():
        if not isinstance(v, dict):
            continue
        n1 = v.get("node1", "")
        n2 = v.get("node2", "")
        if n1 not in name2id or n2 not in name2id:
            continue
        area = float(v.get("area", 1.0))
        ori_rows.append({
            "id": ori_id, "name": name,
            "sj_0": name2id[n1], "sj_1": name2id[n2],
            "A": area, "orientation": "side",
            "z_o": 0.0, "y_max": max(1.0, area ** 0.5),
            "shape": "rect_open", "g1": area ** 0.5, "g2": area ** 0.5, "g3": 0.0,
            "flow_model": "legacy",
            "C": 0.65,
        })
        ori_id += 1

    # --- bleeders（分水口 → 元数据 + 初始流量）---
    bleeders = bd.get("bleeders", {}) or {}
    bleeder_info = []
    # 构建 channel→节点对 映射（用于根据桩号找最近节点）
    _ch_node_map: dict[str, tuple[str, str]] = {}
    for _ch_name, _ch in channels.items():
        _cn1 = _ch.get("node1", "")
        _cn2 = _ch.get("node2", "")
        if _cn1 and _cn2:
            _ch_node_map[_ch_name] = (_cn1, _cn2)
    for bname, b in bleeders.items():
        if not isinstance(b, dict):
            continue
        node_name = b.get("node") or ""
        sj_idx = name2id.get(node_name, -1) if node_name else -1
        # 如果 node 为空但有 channel 信息，用 channel 的上游节点
        if sj_idx < 0 and "channel" in b:
            ch_info = b["channel"]
            if isinstance(ch_info, list) and len(ch_info) >= 1:
                ch_name = ch_info[0]
                ch_nodes = _ch_node_map.get(ch_name)
                if ch_nodes:
                    # 用上游节点作为分水口的关联节点
                    node_name = ch_nodes[0]
                    sj_idx = name2id.get(node_name, -1)
        bleeder_info.append({
            "name": bname,
            "node": node_name,
            "sj": sj_idx,
        })

    # --- pumps（泵站 → 元数据）---
    pumps_raw = bd.get("pumps", {}) or {}
    pump_info = []
    for pname, p in pumps_raw.items():
        if isinstance(p, dict):
            pump_info.append({
                "name": pname,
                "node1": p.get("node1", ""),
                "node2": p.get("node2", ""),
                "curve": p.get("curve", ""),
                "sj_0": name2id.get(p.get("node1", ""), -1),
                "sj_1": name2id.get(p.get("node2", ""), -1),
            })

    # --- pumps DataFrame ---
    curves_dict = bd.get("curves", {}) or {}
    turbines_raw = bd.get("turbines", {}) or {}

    pump_rows: list[dict] = []
    pump_id = 0

    for pname, p in pumps_raw.items():
        if not isinstance(p, dict):
            continue
        n1 = p.get("node1", "")
        n2 = p.get("node2", "")
        condition_range = p.get("conditionRange", [-10.0, 10.0])

        sj_0 = name2id.get(n1)
        sj_1 = name2id.get(n2)
        if sj_0 is None or sj_1 is None:
            continue

        curve_name = p.get("curve", "")
        curve_data = curves_dict.get(curve_name, {}).get("data", []) if curve_name else []

        if curve_data:
            # 泵曲线格式：[[dH, Q, P], ...]
            dh_vals = [float(row[0]) for row in curve_data if len(row) >= 2]
            q_vals  = [float(row[1]) for row in curve_data if len(row) >= 2]
            a_q = max(abs(q) for q in q_vals) if q_vals else 5.0
            a_h = max(abs(dh) for dh in dh_vals) if dh_vals else 10.0
            dH_min = min(dh_vals) if dh_vals else float(condition_range[0])
            dH_max = max(dh_vals) if dh_vals else float(condition_range[1])
        else:
            a_q = 5.0
            a_h = 10.0
            dH_min = float(condition_range[0])
            dH_max = float(condition_range[1])

        pump_rows.append({
            "id": pump_id,
            "name": pname,
            "sj_0": sj_0,
            "sj_1": sj_1,
            "z_p": 0.0,
            "a_q": max(a_q, 0.1),
            "a_h": max(a_h, 0.1),
            "dH_min": dH_min,
            "dH_max": dH_max,
        })
        pump_id += 1

    pumps_df: pd.DataFrame | None = pd.DataFrame(pump_rows) if pump_rows else None

    turbine_info = []
    for turbine_name, turbine in turbines_raw.items():
        if not isinstance(turbine, dict):
            continue
        node1 = turbine.get("node1", "")
        node2 = turbine.get("node2", "")
        if node1 not in name2id or node2 not in name2id:
            continue
        curve_name = turbine.get("curve", "")
        turbine_info.append({
            "name": turbine_name,
            "node1": node1,
            "node2": node2,
            "sj_0": name2id[node1],
            "sj_1": name2id[node2],
            "curve": curve_name,
            "curve_data": curves_dict.get(curve_name, {}).get("data", []) if curve_name else [],
        })

    boundaries = init_data.get("boundaries", {})
    init_bleeders = init_data.get("bleeders", {})
    init_gates = init_data.get("gates", {})

    time_series_raw = model_obj.get("timeSeries", {})

    # 解析时序数据：将原始 WXQ timeSeries 格式转为结构化 dict
    time_series_parsed: dict = {}
    series_link = time_series_raw.get("seriesLink", {})
    series_data = time_series_raw.get("seriesData", {})
    for series_name, link_info in series_link.items():
        if not isinstance(link_info, (list, tuple)) or len(link_info) < 2:
            continue
        target_type = str(link_info[0])   # "nodes", "pumps", "gates", etc.
        target_name = str(link_info[1])   # 目标设备/节点名称
        data_points = series_data.get(series_name, [])
        if not data_points:
            continue
        times_s = [float(pt[0]) for pt in data_points]
        values = [float(pt[1]) for pt in data_points]
        time_series_parsed[series_name] = {
            "target_type": target_type,
            "target_name": target_name,
            "times_s": times_s,
            "values": values,
            "values_by_time": dict(zip(times_s, values)),
        }

    time_control = model_obj.get("timeControl", {})

    return {
        "superjunctions": pd.DataFrame(sj_rows),
        "superlinks": pd.DataFrame(sl_rows),
        "links": links_df,
        "junctions": junctions_df,
        "transects": transects,
        "section_geometry_mode": section_geometry_mode,
        "section_geometry_audit": section_geometry_audit,
        "section_geometry_fallbacks": [
            row for row in section_geometry_audit if bool(row.get("fallback"))
        ],
        "section_geometry_issues": section_geometry_issues,
        "channel_discretization": channel_discretization,
        "orifices": pd.DataFrame(ori_rows) if ori_rows else None,
        "turbines": turbine_info,
        "model_name": model_name,
        "timeControl": time_control,
        "time_control": time_control,
        "bleeders": bleeder_info,
        "pumps": pump_info,
        "pumps_df": pumps_df,
        "siphons_info": siphon_info,
        "siphon_from_channel": siphon_from_channel,
        "valve_curves": bd.get("curves", {}),
        "boundaries": boundaries,
        "initial_data": {
            "channels": init_channels,
            "bleeders": init_bleeders,
            "gates": init_gates,
            "turbines": init_data.get("turbines", {}),
        },
        "time_series": time_series_parsed,
        "topology_repairs": topology_repairs,
    }


# ── _haversine: 经纬度距离（米）──────────────────────────────────────────

def _haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """经纬度距离（米）。"""
    R = 6_371_000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _coordinate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """兼容经纬度坐标和以米为单位的平面投影坐标。"""
    is_lon_lat = all((abs(x) <= 180 and abs(y) <= 90) for x, y in ((x1, y1), (x2, y2)))
    if is_lon_lat:
        return _haversine(x1, y1, x2, y2)
    return math.hypot(x2 - x1, y2 - y1)


__all__ = ["parse_wxq_json", "turbine_discharge_from_curve", "_haversine", "_coordinate_distance"]


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2:
        path = Path(sys.argv[1])
        result = parse_wxq_json(path)
        print(f"model_name: {result.get('model_name')}")
        print(f"superjunctions: {len(result.get('superjunctions', []))}")
        print(f"superlinks: {len(result.get('superlinks', []))}")
        ori = result.get('orifices')
        print(f"orifices: {len(ori) if ori is not None else 0}")
        print(f"siphons_info: {len(result.get('siphons_info', []))}")
        print(f"bleeders: {len(result.get('bleeders', []))}")
        print(f"pumps: {len(result.get('pumps', []))}")
    else:
        print("Usage: python3 wxq_to_pipedream_minimal.py <path/to/model.json>")
