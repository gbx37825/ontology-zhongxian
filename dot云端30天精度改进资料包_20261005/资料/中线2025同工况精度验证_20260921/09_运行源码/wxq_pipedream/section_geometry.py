"""WXQ cross-section geometry preservation and Pipedream serialization."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np


SectionKind = Literal["rect_open", "trapezoidal", "irregular"]


@dataclass(frozen=True)
class SectionGeometry:
    source_name: str | None
    z_inv: float
    height: float
    bottom_width: float
    side_slope: float
    raw_yz: tuple[tuple[float, float], ...]
    shape_kind: SectionKind

    def analytic_geometry(self) -> dict[str, float]:
        return {
            "z_inv": self.z_inv,
            "height": self.height,
            "bottom_width": self.bottom_width,
            "side_slope": self.side_slope,
        }


def raw_yz_points(yz: list | tuple) -> list[tuple[float, float]]:
    """Filter invalid points while preserving point order and vertical walls."""
    result: list[tuple[float, float]] = []
    for point in yz or []:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            station = float(point[0])
            elevation = float(point[1])
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(station) and math.isfinite(elevation)):
            continue
        candidate = (station, elevation)
        if not result or candidate != result[-1]:
            result.append(candidate)
    return result


def _canonical_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(points) < 2:
        return points
    result = list(points)
    if result[0][0] > result[-1][0]:
        result.reverse()
    return result


def _analytic_contour(
    z_inv: float,
    height: float,
    bottom_width: float,
    side_slope: float,
) -> list[tuple[float, float]]:
    side_run = side_slope * height
    return [
        (0.0, z_inv + height),
        (side_run, z_inv),
        (side_run + bottom_width, z_inv),
        (2.0 * side_run + bottom_width, z_inv + height),
    ]


def _fallback_parameters(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    stations = [point[0] for point in points]
    elevations = [point[1] for point in points]
    z_inv = min(elevations)
    height = max(elevations) - z_inv
    z_tolerance = max(1e-6, height * 1e-6)
    bottom_stations = [
        station for station, elevation in points if abs(elevation - z_inv) <= z_tolerance
    ]
    bottom_width = (
        max(bottom_stations) - min(bottom_stations)
        if len(bottom_stations) >= 2
        else max(stations) - min(stations)
    )
    top_width = max(stations) - min(stations)
    side_slope = max((top_width - bottom_width) / max(2.0 * height, 1e-12), 0.0)
    return z_inv, max(height, 0.01), max(bottom_width, 0.01), side_slope


def _clip_polygon_below(
    polygon: list[tuple[float, float]],
    elevation: float,
) -> list[tuple[float, float]]:
    if not polygon:
        return []
    output: list[tuple[float, float]] = []
    previous = polygon[-1]
    previous_inside = previous[1] <= elevation
    for current in polygon:
        current_inside = current[1] <= elevation
        if current_inside != previous_inside:
            dz = current[1] - previous[1]
            fraction = 0.0 if abs(dz) <= 1e-15 else (elevation - previous[1]) / dz
            output.append(
                (
                    previous[0] + fraction * (current[0] - previous[0]),
                    elevation,
                )
            )
        if current_inside:
            output.append(current)
        previous = current
        previous_inside = current_inside
    return output


def _polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    return abs(
        sum(
            left[0] * right[1] - right[0] * left[1]
            for left, right in zip(points, points[1:] + points[:1])
        )
    ) / 2.0


def area_below(points: list[tuple[float, float]], elevation: float) -> float:
    return _polygon_area(_clip_polygon_below(points, elevation))


def _segments_intersect(
    first_start: tuple[float, float],
    first_end: tuple[float, float],
    second_start: tuple[float, float],
    second_end: tuple[float, float],
) -> bool:
    def orientation(a, b, c) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    o1 = orientation(first_start, first_end, second_start)
    o2 = orientation(first_start, first_end, second_end)
    o3 = orientation(second_start, second_end, first_start)
    o4 = orientation(second_start, second_end, first_end)
    tolerance = 1e-12
    return o1 * o2 < -tolerance and o3 * o4 < -tolerance


def _has_self_intersection(points: list[tuple[float, float]]) -> bool:
    segments = list(zip(points, points[1:]))
    for first_index, first in enumerate(segments):
        for second_index in range(first_index + 2, len(segments)):
            if second_index == first_index + 1:
                continue
            if _segments_intersect(first[0], first[1], segments[second_index][0], segments[second_index][1]):
                return True
    return False


def classify_section(
    yz: list | tuple,
    *,
    source_name: str | None = None,
) -> SectionGeometry | None:
    points = _canonical_points(raw_yz_points(yz))
    if len(points) < 3 or _has_self_intersection(points):
        return None
    stations = np.asarray([point[0] for point in points], dtype=float)
    elevations = np.asarray([point[1] for point in points], dtype=float)
    width = float(stations.max() - stations.min())
    z_inv = float(elevations.min())
    height = float(elevations.max() - z_inv)
    if width <= 0.0 or height <= 0.0:
        return None

    fallback_z, fallback_h, fallback_b, fallback_m = _fallback_parameters(points)
    shape_kind: SectionKind = "irregular"
    z_tolerance = max(1e-6, height * 1e-6)
    x_tolerance = max(1e-6, width * 1e-5)
    bottom = [point for point in points if abs(point[1] - z_inv) <= z_tolerance]
    top = [point for point in points if abs(point[1] - (z_inv + height)) <= z_tolerance]
    if len(bottom) >= 2 and len(top) >= 2:
        bottom_left = min(point[0] for point in bottom)
        bottom_right = max(point[0] for point in bottom)
        top_left = min(point[0] for point in top)
        top_right = max(point[0] for point in top)
        bottom_width = bottom_right - bottom_left
        left_slope = (bottom_left - top_left) / height
        right_slope = (top_right - bottom_right) / height
        slope_tolerance = max(1e-6, max(abs(left_slope), abs(right_slope), 1.0) * 1e-3)
        if (
            bottom_width > 0.0
            and left_slope >= -slope_tolerance
            and right_slope >= -slope_tolerance
            and abs(left_slope - right_slope) <= slope_tolerance
        ):
            side_slope = max((left_slope + right_slope) / 2.0, 0.0)
            all_on_boundary = True
            for station, elevation in points:
                relative_z = elevation - z_inv
                if relative_z <= z_tolerance:
                    if station < bottom_left - x_tolerance or station > bottom_right + x_tolerance:
                        all_on_boundary = False
                        break
                    continue
                expected_left = bottom_left - side_slope * relative_z
                expected_right = bottom_right + side_slope * relative_z
                if min(abs(station - expected_left), abs(station - expected_right)) > x_tolerance:
                    all_on_boundary = False
                    break
            if all_on_boundary:
                analytic = _analytic_contour(z_inv, height, bottom_width, side_slope)
                area_ok = True
                for fraction in (0.25, 0.5, 0.75, 1.0):
                    level = z_inv + fraction * height
                    source_area = area_below(points, level)
                    analytic_area = area_below(analytic, level)
                    if source_area > 1e-9 and abs(analytic_area - source_area) / source_area > 0.01:
                        area_ok = False
                        break
                if area_ok:
                    fallback_z = z_inv
                    fallback_h = height
                    fallback_b = bottom_width
                    fallback_m = side_slope
                    shape_kind = "rect_open" if side_slope <= slope_tolerance else "trapezoidal"

    return SectionGeometry(
        source_name=source_name,
        z_inv=fallback_z,
        height=fallback_h,
        bottom_width=fallback_b,
        side_slope=fallback_m,
        raw_yz=tuple(points),
        shape_kind=shape_kind,
    )


def _arc_profile(section: SectionGeometry) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = _canonical_points(list(section.raw_yz))
    coordinates = np.asarray(points, dtype=float)
    coordinates[:, 0] -= coordinates[:, 0].min()
    coordinates[:, 1] -= coordinates[:, 1].min()
    segment_lengths = np.sqrt(np.sum(np.diff(coordinates, axis=0) ** 2, axis=1))
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    if cumulative[-1] <= 0.0:
        raise ValueError("断面轮廓累计弧长必须大于0")
    return cumulative / cumulative[-1], coordinates[:, 0], coordinates[:, 1]


def interpolate_section_geometry(
    start: SectionGeometry,
    end: SectionGeometry,
    fraction: float,
) -> SectionGeometry:
    fraction = min(max(float(fraction), 0.0), 1.0)
    if fraction <= 0.0:
        return start
    if fraction >= 1.0:
        return end
    z_inv = start.z_inv + (end.z_inv - start.z_inv) * fraction

    analytic_kinds = {"rect_open", "trapezoidal"}
    if start.shape_kind in analytic_kinds and end.shape_kind in analytic_kinds:
        height = start.height + (end.height - start.height) * fraction
        bottom_width = start.bottom_width + (end.bottom_width - start.bottom_width) * fraction
        side_slope = start.side_slope + (end.side_slope - start.side_slope) * fraction
        shape_kind: SectionKind = "rect_open" if side_slope <= 1e-9 else "trapezoidal"
        return SectionGeometry(
            source_name=None,
            z_inv=z_inv,
            height=height,
            bottom_width=bottom_width,
            side_slope=side_slope,
            raw_yz=tuple(_analytic_contour(z_inv, height, bottom_width, side_slope)),
            shape_kind=shape_kind,
        )

    start_s, start_x, start_y = _arc_profile(start)
    end_s, end_x, end_y = _arc_profile(end)
    samples = np.unique(np.concatenate((start_s, end_s)))
    if samples.size > 1000:
        samples = np.linspace(0.0, 1.0, 1000)
    x_values = np.interp(samples, start_s, start_x) + (
        np.interp(samples, end_s, end_x) - np.interp(samples, start_s, start_x)
    ) * fraction
    y_values = np.interp(samples, start_s, start_y) + (
        np.interp(samples, end_s, end_y) - np.interp(samples, start_s, start_y)
    ) * fraction
    raw = [(float(x_value), float(z_inv + y_value)) for x_value, y_value in zip(x_values, y_values)]
    classified = classify_section(raw)
    if classified is None:
        raise ValueError("过渡断面无有效宽度、高度或存在自交")
    if classified.shape_kind in analytic_kinds:
        return classified
    fallback_z, fallback_h, fallback_b, fallback_m = _fallback_parameters(raw)
    return SectionGeometry(
        source_name=None,
        z_inv=fallback_z,
        height=fallback_h,
        bottom_width=fallback_b,
        side_slope=fallback_m,
        raw_yz=tuple(raw),
        shape_kind="irregular",
    )


def _single_valued_transect(
    section: SectionGeometry,
    *,
    horiz_points: int = 1000,
    vert_points: int = 100,
) -> dict[str, list[float] | int]:
    points = _canonical_points(list(section.raw_yz))
    x_values = np.asarray([point[0] for point in points], dtype=float)
    y_values = np.asarray([point[1] for point in points], dtype=float)
    x_values -= x_values.min()
    y_values -= y_values.min()
    width = float(x_values.max())
    if width <= 0.0 or float(y_values.max()) <= 0.0:
        raise ValueError("不规则断面宽度和高度必须大于0")
    x_tolerance = max(1e-12, width * 1e-12)
    if np.any(np.diff(x_values) < -x_tolerance):
        raise ValueError("不规则断面存在横向回折，无法转换为单值横坐标")
    duplicates = np.flatnonzero(np.diff(x_values) <= x_tolerance)
    if duplicates.size:
        epsilon = width / max(int(horiz_points) - 1, 1)
        adjusted = x_values.copy()
        index = 0
        while index < len(adjusted):
            end = index
            while end + 1 < len(adjusted) and abs(x_values[end + 1] - x_values[index]) <= x_tolerance:
                end += 1
            count = end - index + 1
            if count > 1:
                if index == 0:
                    values = x_values[index] + epsilon * np.arange(count)
                elif end == len(adjusted) - 1:
                    values = x_values[index] - epsilon * np.arange(count - 1, -1, -1)
                else:
                    center = x_values[index]
                    values = center + epsilon * (np.arange(count) - (count - 1) / 2.0)
                lower = adjusted[index - 1] + epsilon if index else 0.0
                upper = x_values[end + 1] - epsilon if end + 1 < len(adjusted) else width
                if values[0] < lower or values[-1] > upper:
                    values = np.linspace(lower, upper, count)
                adjusted[index : end + 1] = values
            index = end + 1
        x_values = adjusted
    if np.any(np.diff(x_values) <= 0.0):
        raise ValueError("不规则断面最终横坐标未严格递增")
    return {
        "x": x_values.tolist(),
        "y": y_values.tolist(),
        "horiz_points": min(max(int(horiz_points), 2), 1000),
        "vert_points": min(max(int(vert_points), 2), 1000),
    }


def _validate_irregular(
    section: SectionGeometry,
    transect: dict[str, list[float] | int],
    *,
    height_retention_min: float = 0.95,
    area_relative_tolerance: float = 0.01,
) -> dict:
    target_points = _canonical_points(list(section.raw_yz))
    converted_points = list(zip(transect["x"], transect["y"]))
    converted_height = max(transect["y"]) - min(transect["y"])
    height_ratio = converted_height / max(section.height, 1e-12)
    area_errors = {}
    passed = height_ratio >= height_retention_min
    for fraction in (0.25, 0.5, 0.75, 1.0):
        target_level = section.z_inv + fraction * section.height
        target_area = area_below(target_points, target_level)
        converted_area = area_below(converted_points, fraction * section.height)
        relative_error = (
            abs(converted_area - target_area) / target_area if target_area > 1e-9 else 0.0
        )
        area_errors[str(fraction)] = relative_error
        passed = passed and relative_error <= area_relative_tolerance
    passed = passed and all(
        right > left for left, right in zip(transect["x"], transect["x"][1:])
    )
    return {
        "passed": bool(passed),
        "height_ratio": float(height_ratio),
        "area_relative_errors": area_errors,
    }


def to_pipedream_geometry(
    section: SectionGeometry,
    *,
    horiz_points: int = 1000,
) -> dict:
    if section.shape_kind in {"rect_open", "trapezoidal"}:
        return {
            "shape": section.shape_kind,
            "g1": section.height,
            "g2": section.bottom_width,
            "g3": 0.0 if section.shape_kind == "rect_open" else section.side_slope,
            "transect": None,
            "quality": {
                "passed": True,
                "fallback": False,
                "height_ratio": 1.0,
                "area_relative_errors": {"0.25": 0.0, "0.5": 0.0, "0.75": 0.0, "1.0": 0.0},
            },
        }
    try:
        transect = _single_valued_transect(section, horiz_points=horiz_points)
        quality = _validate_irregular(section, transect)
    except ValueError as exc:
        transect = None
        quality = {"passed": False, "reason": str(exc)}
    if transect is not None and quality["passed"]:
        quality["fallback"] = False
        return {
            "shape": "irregular",
            "g1": section.height,
            "g2": section.bottom_width,
            "g3": section.side_slope,
            "transect": transect,
            "quality": quality,
        }
    quality["fallback"] = True
    quality.setdefault("reason", "不规则断面高度或分层面积质量门未通过")
    return {
        "shape": "trapezoidal",
        "g1": section.height,
        "g2": section.bottom_width,
        "g3": section.side_slope,
        "transect": None,
        "quality": quality,
    }
