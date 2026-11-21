"""Stiffened-gas equation of state and Euler fluxes.

The conservative variables are ``U = [rho, rho*u, E]`` with

    p = (gamma - 1) * (E - 0.5 * rho * u**2) - gamma * p_inf.

All routines broadcast over leading dimensions, so ``U`` may have shape
``(3,)``, ``(3, nx)`` or ``(3, nx, n_phase)``.
"""

from __future__ import annotations

import numpy as np


class StiffenedGas:
    """Ideal/stiffened gas with constant ``gamma`` and ``p_inf``."""

    def __init__(self, gamma: float = 1.4, p_inf: float = 0.0):
        self.gamma = float(gamma)
        self.p_inf = float(p_inf)
        if self.gamma <= 1.0:
            raise ValueError("gamma must be greater than 1")

    # ------------------------------------------------------------------
    # primitive / conservative conversion
    # ------------------------------------------------------------------
    def primitive(self, U):
        """Return ``(rho, u, p)`` from conservative variables."""
        U = np.asarray(U, dtype=float)
        rho = np.maximum(U[0], 1.0e-14)
        u = U[1] / rho
        p = (self.gamma - 1.0) * (U[2] - 0.5 * rho * u * u) - self.gamma * self.p_inf
        return rho, u, np.maximum(p, 1.0e-14)

    def conservative(self, rho, u, p):
        """Return conservative variables from primitive variables."""
        rho = np.asarray(rho, dtype=float)
        u = np.asarray(u, dtype=float)
        p = np.asarray(p, dtype=float)
        E = (p + self.gamma * self.p_inf) / (self.gamma - 1.0) + 0.5 * rho * u * u
        return np.stack((rho, rho * u, E), axis=0)

    def sound_speed(self, U):
        rho, _, p = self.primitive(U)
        return np.sqrt(self.gamma * (p + self.p_inf) / rho)

    def specific_enthalpy(self, U):
        rho, u, p = self.primitive(U)
        return (U[2] + p) / rho

    def max_speed(self, U):
        _, u, _ = self.primitive(U)
        return np.max(np.abs(u) + self.sound_speed(U))

    # ------------------------------------------------------------------
    # fluxes
    # ------------------------------------------------------------------
    def flux(self, U):
        """Physical Euler flux."""
        U = np.asarray(U, dtype=float)
        rho, u, p = self.primitive(U)
        F = np.empty_like(U)
        F[0] = U[1]
        F[1] = U[1] * u + p
        F[2] = (U[2] + p) * u
        return F


def as_flux_shape(F, ndim):
    """Broadcast a flux array to the working dimension."""
    F = np.asarray(F, dtype=float)
    while F.ndim < ndim:
        F = F[..., None]
    return F
