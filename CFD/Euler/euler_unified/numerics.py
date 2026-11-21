"""Shared numerical tools: limiters, Riemann fluxes and exact solutions."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .eos import StiffenedGas


# ----------------------------------------------------------------------
# limiters
# ----------------------------------------------------------------------
def minmod(a, b, c, M=0.0, dx=1.0):
    """Vectorised minmod with the TVB modification used in the DG code.

    ``a`` is the candidate slope; ``b`` and ``c`` are the one-sided
    differences.  For ``M=0`` this is the classical minmod limiter.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    bc_min = np.minimum(b * np.sign(a), c * np.sign(a))
    small = np.abs(a) <= M * dx * dx
    limited = np.sign(a) * np.maximum(0.0, np.minimum(np.abs(a), bc_min))
    return np.where(small, a, limited)


def minmod2(a, b):
    """Classical two-argument minmod for finite-volume slopes."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.where(a * b > 0.0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


# ----------------------------------------------------------------------
# numerical fluxes
# ----------------------------------------------------------------------
def numerical_flux(UL, UR, gas: StiffenedGas, kind: str = "llf"):
    """Numerical flux for the 1-D Euler equations.

    Parameters
    ----------
    UL, UR : ndarray, shape (3, n_face)
        Left and right states at every face.
    gas : StiffenedGas
        Equation of state used to evaluate physical fluxes.
    kind : {"llf", "rusanov", "hll", "hllc"}
    """
    kind = kind.lower()
    if kind == "rusanov":
        kind = "llf"

    UL = np.asarray(UL, dtype=float)
    UR = np.asarray(UR, dtype=float)
    FL = gas.flux(UL)
    FR = gas.flux(UR)
    rhoL, uL, pL = gas.primitive(UL)
    rhoR, uR, pR = gas.primitive(UR)
    aL = gas.sound_speed(UL)
    aR = gas.sound_speed(UR)

    if kind == "llf":
        alpha = np.maximum(np.abs(uL) + aL, np.abs(uR) + aR)
        return 0.5 * (FL + FR) - 0.5 * alpha * (UR - UL)

    # Davis wave-speed estimates, used by both HLL and HLLC.
    SL = np.minimum(uL - aL, uR - aR)
    SR = np.maximum(uL + aL, uR + aR)

    if kind == "hll":
        denom = np.where(np.abs(SR - SL) < 1.0e-14, 1.0e-14, SR - SL)
        FHLL = (SR * FL - SL * FR + SL * SR * (UR - UL)) / denom
        F = np.where(SL >= 0.0, FL, np.where(SR <= 0.0, FR, FHLL))
        return F

    if kind != "hllc":
        raise ValueError(f"unknown flux: {kind!r}")

    # HLLC.  Compute all branches and select with masks; denominators are
    # protected so that no warning is emitted for inactive branches.
    denom_star = rhoL * (SL - uL) - rhoR * (SR - uR)
    denom_star = np.where(np.abs(denom_star) < 1.0e-14, 1.0e-14, denom_star)
    SM = (pR - pL + rhoL * uL * (SL - uL) - rhoR * uR * (SR - uR)) / denom_star

    def star_state(U, rho, u, p, S, SM):
        def _safe(d):
            # Keep the sign: S_L - u_L is negative by construction.
            return np.where(np.abs(d) < 1.0e-14,
                            np.where(d >= 0.0, 1.0e-14, -1.0e-14), d)
        rho_star = rho * (S - u) / _safe(S - SM)
        e_star = U[2] / rho + (SM - u) * (SM + p / (rho * _safe(S - u)))
        return np.stack((rho_star, rho_star * SM, rho_star * e_star), axis=0)

    UstarL = star_state(UL, rhoL, uL, pL, SL, SM)
    UstarR = star_state(UR, rhoR, uR, pR, SR, SM)
    FstarL = FL + SL * (UstarL - UL)
    FstarR = FR + SR * (UstarR - UR)

    F = np.where(SL >= 0.0, FL,
                 np.where(SR <= 0.0, FR,
                          np.where(SM >= 0.0, FstarL, FstarR)))
    # Degenerate fallback when a Riemann fan collapses.
    bad = (~np.isfinite(F)).any(axis=0)
    if np.any(bad):
        alpha = np.maximum(np.abs(uL) + aL, np.abs(uR) + aR)
        Fllf = 0.5 * (FL + FR) - 0.5 * alpha * (UR - UL)
        F = np.where(bad[None, :], Fllf, F)
    return F


# ----------------------------------------------------------------------
# exact solution of the single-interface Riemann problem
# ----------------------------------------------------------------------
_RIEMANN_PATH_ADDED = False


def _ensure_riemann_path():
    global _RIEMANN_PATH_ADDED
    if _RIEMANN_PATH_ADDED:
        return
    path = Path(__file__).resolve().parents[2] / "Riemann" / "code"
    if not path.is_dir():
        raise RuntimeError(f"cannot find the exact Riemann solver at {path}")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    _RIEMANN_PATH_ADDED = True


def exact_riemann(left, right, x, t, x_interface):
    """Sample the exact Riemann solution on the grid ``x``.

    ``left`` and ``right`` are dictionaries with keys ``rho``, ``u``, ``p``,
    ``gamma`` and optionally ``p_inf``.  The initial discontinuity is at
    ``x_interface``.
    """
    _ensure_riemann_path()
    import Fluid as Fdexact
    import Exact_Riemann_solver as ERS
    import Exact_Riemann_complete_solve as ERCS

    x = np.asarray(x, dtype=float)
    WL = Fdexact.Fluid(left["rho"], left["u"], left["p"],
                       left["gamma"], left.get("p_inf", 0.0))
    WR = Fdexact.Fluid(right["rho"], right["u"], right["p"],
                       right["gamma"], right.get("p_inf", 0.0))
    solver = ERS.ExactRiemannSolver(WL, WR, 1.0e-12, 1000)
    p_star, u_star = solver.solver()

    # The complete solver samples [x_L, x_R] uniformly.  Our grids are
    # uniform, so N = len(x) - 1 reproduces the same abscissae after the
    # interface shift.
    x_L = float(x[0] - x_interface)
    x_R = float(x[-1] - x_interface)
    complete = ERCS.Riemann_Complete_solve(
        p_star, u_star, WL, WR, t, x_L, x_R, max(len(x) - 1, 1)
    )
    xs, rho, u, p = complete.solve()
    # Guard against tiny round-off differences in the abscissae.
    if len(xs) != len(x) or not np.allclose(xs + x_interface, x):
        # Fall back to interpolation on the exact solution.
        order = np.argsort(xs)
        rho = np.interp(x - x_interface, xs[order], rho[order])
        u = np.interp(x - x_interface, xs[order], u[order])
        p = np.interp(x - x_interface, xs[order], p[order])
    return rho, u, p
