"""Solver selection and the tested standard-to-Numba parameter convention."""

from __future__ import annotations

import numpy as np

DEFAULT_BACKEND = "numba"
BACKENDS = ("numba", "standard")
COMPATIBILITY_REVISIONS = {
    "numba": "standard-equivalent-numba-v1",
    "standard": "bundled-standard-v1",
}
RESTART_GUARD_ARRAYS = (
    "nk", "_ki", "_link_end", "_k_1k", "_C_uk", "_C_dk", "_C_ik", "_Co",
)


def solver_class(backend):
    if backend == "standard":
        from pipedream_solver.superlink import SuperLink

        return SuperLink
    if backend != "numba":
        raise ValueError(f"Unknown solver backend: {backend}; choose {BACKENDS}")
    try:
        from numba import config

        if config.DISABLE_JIT:
            raise RuntimeError("Numba JIT is disabled; unset NUMBA_DISABLE_JIT or use --backend standard")
        from pipedream_solver.nsuperlink import nSuperLink
    except ImportError as exc:
        raise RuntimeError("Numba is unavailable; run uv sync or explicitly use --backend standard") from exc
    return nSuperLink


def adapt_numba_model(model):
    """Adapt once after construction, never after restoring an adapted state."""
    if model._bc_method != "z" or np.any(model._Sf_method_ik != 0):
        raise ValueError("Numba equivalence requires z end conditions and Manning friction")
    counts = np.asarray(model.nk)
    if counts.shape != (model.NK,) or (counts < 1).any():
        raise ValueError("Invalid number of internal links for Numba LSQ")
    offsets = np.r_[0, np.cumsum(counts[:-1] - 1)].astype(np.int64)
    groups = model._ki[~model._link_end]
    expected_groups = np.repeat(np.arange(model.NK), counts - 1)
    if not np.array_equal(groups, expected_groups):
        raise ValueError("Numba LSQ requires contiguous internal links grouped by superlink")

    converted, losses = {}, {}
    for name in ("_C_uk", "_C_dk", "_C_ik"):
        original = getattr(model, name)
        if not np.isfinite(original).all() or (original <= 0).any():
            raise ValueError(f"Local-loss parameter must be finite and positive: {name}")
        # Standard divides by C**2; Numba multiplies by the local-loss factor.
        with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
            equivalent = 1.0 / np.square(original)
        if not np.isfinite(equivalent).all() or (equivalent <= 0).any():
            raise ValueError(f"Local-loss parameter conversion overflows: {name}")
        converted[name] = equivalent
        losses[name] = {
            "standard_min": float(original.min()), "standard_max": float(original.max()),
            "numba_min": float(equivalent.min()), "numba_max": float(equivalent.max()),
            "changed_count": int(np.count_nonzero(original != equivalent)),
        }
    changed = model._k_1k != offsets
    diagnostics = {
        "lsq_offset_changed_segments": int(changed.sum()),
        "lsq_offset_changed_segments_with_internal_nodes": int((changed & (counts > 1)).sum()),
        "local_loss_parameters": losses,
        "orifice_discharge_coefficients_unchanged": True,
    }
    model._k_1k = offsets
    for name, values in converted.items():
        setattr(model, name, values)
    return diagnostics


def create_solver(backend=DEFAULT_BACKEND, **kwargs):
    model = solver_class(backend)(**kwargs)
    diagnostics = adapt_numba_model(model) if backend == "numba" else {}
    return model, diagnostics
