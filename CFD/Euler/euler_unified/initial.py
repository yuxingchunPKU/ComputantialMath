"""Initial and exact reference fields for the Euler test cases."""

from __future__ import annotations

import numpy as np

from .numerics import exact_riemann


def primitive_at(cfg: dict, x):
    """Evaluate the initial primitive state at ``x``."""
    x = np.asarray(x, dtype=float)
    initial = cfg["initial"]
    itype = initial["type"]
    if itype == "smooth":
        rho = initial["rho0"] + initial["rho_amp"] * np.sin(initial["k"] * np.pi * x)
        u = np.full_like(x, initial["u0"])
        p = np.full_like(x, initial["p0"])
        return rho, u, p

    left = initial["left"]
    right = initial["right"]
    if itype == "constant":
        return (np.full_like(x, left["rho"]),
                np.full_like(x, left["u"]),
                np.full_like(x, left["p"]))
    if itype == "sod":
        mask = x < initial["x_interface"]
        rho = np.where(mask, left["rho"], right["rho"])
        u = np.where(mask, left["u"], right["u"])
        p = np.where(mask, left["p"], right["p"])
        return rho, u, p
    raise ValueError(f"unsupported initial type: {itype}")


def _left_right_fluids(cfg: dict):
    if cfg["medium"] == "single":
        fluid = cfg["fluids"]
        left = {**cfg["initial"]["left"], **fluid}
        right = {**cfg["initial"]["right"], **fluid}
    else:
        left = {**cfg["initial"]["left"], **cfg["fluids"]["left"]}
        right = {**cfg["initial"]["right"], **cfg["fluids"]["right"]}
    return left, right


def exact_fields(cfg: dict, x, t: float):
    """Return the exact/reference ``(rho, u, p)`` on ``x`` at time ``t``."""
    x = np.asarray(x, dtype=float)
    initial = cfg["initial"]
    if initial["type"] == "smooth":
        rho = initial["rho0"] + initial["rho_amp"] * np.sin(
            initial["k"] * np.pi * (x - initial["u0"] * t)
        )
        u = np.full_like(x, initial["u0"])
        p = np.full_like(x, initial["p0"])
        return rho, u, p

    left, right = _left_right_fluids(cfg)
    return exact_riemann(left, right, x, t, initial["x_interface"])
