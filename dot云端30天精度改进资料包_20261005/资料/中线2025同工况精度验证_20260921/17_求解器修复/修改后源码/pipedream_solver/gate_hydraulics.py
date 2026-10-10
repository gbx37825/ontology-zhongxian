"""Hydrostatic pressure integration for explicitly rectangular side gates.

Not a critical-flow/channel model or a replacement for calibrated contraction
coefficients. C is supplied by the caller. Integrating the submerged and free
parts of the aperture avoids dropping tailwater when the gate clears the water.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def rectangular_gate_discharge(hu, hd, opening, width, coefficient, oneway=False):
    if opening <= 0 or width <= 0 or (oneway and hd >= hu):
        return 0.0
    high = max(hu, hd)
    if high <= 0:
        return 0.0
    low = max(min(hu, hd), 0.0)
    aperture = min(opening, high)
    wet = min(aperture, low)
    difference = max(high - low, 0.0)
    integral = (wet * np.sqrt(difference)
                + (2.0 / 3.0) * ((high - wet)**1.5 - (high - aperture)**1.5))
    sign = 1.0 if hu > hd else (-1.0 if hu < hd else 0.0)
    return sign * coefficient * width * np.sqrt(2.0 * 9.81) * integral


@njit(cache=True)
def rectangular_gate_terms(ids, heads, upstream, downstream, bed, offsets,
                           heights, widths, coefficients, oneway, controls,
                           flows, alpha, beta, chi, linearize):
    for k in ids:
        z = bed[upstream[k]] + offsets[k]
        hu = heads[upstream[k]] - z
        hd = heads[downstream[k]] - z
        opening = heights[k] * controls[k]
        q = rectangular_gate_discharge(hu, hd, opening, widths[k], coefficients[k], oneway[k])
        flows[k] = q
        if linearize:
            # Central secants keep the near-equal-head derivative finite. The
            # 10 micrometre perturbation does not alter the discharge relation.
            eps = 1e-5
            du = (rectangular_gate_discharge(hu + eps, hd, opening, widths[k], coefficients[k], oneway[k])
                  - rectangular_gate_discharge(hu - eps, hd, opening, widths[k], coefficients[k], oneway[k])) / (2 * eps)
            dd = (rectangular_gate_discharge(hu, hd + eps, opening, widths[k], coefficients[k], oneway[k])
                  - rectangular_gate_discharge(hu, hd - eps, opening, widths[k], coefficients[k], oneway[k])) / (2 * eps)
            alpha[k] = du
            beta[k] = dd
            chi[k] = q - du * heads[upstream[k]] - dd * heads[downstream[k]]
