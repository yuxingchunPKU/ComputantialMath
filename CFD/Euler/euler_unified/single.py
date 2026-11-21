"""Single-medium finite-volume and discontinuous-Galerkin Euler solvers."""

from __future__ import annotations

import numpy as np
from scipy.special import roots_legendre

from .eos import StiffenedGas
from .initial import exact_fields, primitive_at
from .numerics import minmod, minmod2, numerical_flux


def _reflect(U):
    """Reflect a conservative state at a solid wall."""
    out = U.copy()
    out[1] = -out[1]
    return out


class _Mesh:
    def __init__(self, cfg):
        mesh = cfg["mesh"]
        self.x_min = mesh["x_min"]
        self.x_max = mesh["x_max"]
        self.nx = mesh["nx"]
        self.dx = (self.x_max - self.x_min) / self.nx
        self.edges = np.linspace(self.x_min, self.x_max, self.nx + 1)
        self.centers = 0.5 * (self.edges[:-1] + self.edges[1:])
        self.boundary = mesh["boundary"]


class FVSolver:
    """Finite-volume solver (MUSCL/minmod for order 2)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.mesh = _Mesh(cfg)
        self.gas = StiffenedGas(cfg["fluids"]["gamma"], cfg["fluids"]["p_inf"])
        self.order = cfg["order"]
        self.flux_kind = cfg["flux"]
        self.U = self._initial_data()

    def _initial_data(self):
        rho, u, p = primitive_at(self.cfg, self.mesh.centers)
        return self.gas.conservative(rho, u, p)

    # ------------------------------------------------------------------
    def _extended(self, U):
        """Two ghost cells consistent with the selected boundary condition."""
        n = U.shape[1]
        Ue = np.empty((3, n + 2), dtype=float)
        Ue[:, 1:-1] = U
        bc = self.mesh.boundary
        if bc == "periodic":
            Ue[:, 0] = U[:, -1]
            Ue[:, -1] = U[:, 0]
        elif bc == "transmissive":
            Ue[:, 0] = U[:, 0]
            Ue[:, -1] = U[:, -1]
        elif bc == "reflective":
            Ue[:, 0] = _reflect(U[:, 0])
            Ue[:, -1] = _reflect(U[:, -1])
        else:  # pragma: no cover - validated earlier
            raise ValueError(bc)
        return Ue

    def _face_states(self, U):
        Ue = self._extended(U)
        n = U.shape[1]
        # Slopes for the extended cells.  Ghost slopes are copied from the
        # adjacent physical cell; this gives a consistent one-sided boundary
        # reconstruction.
        if self.order == 1:
            s = np.zeros_like(Ue)
        else:
            dL = Ue[:, 1:-1] - Ue[:, :-2]
            dR = Ue[:, 2:] - Ue[:, 1:-1]
            s = np.zeros_like(Ue)
            s[:, 1:-1] = minmod2(dL, dR)
            s[:, 0] = s[:, 1]
            s[:, -1] = s[:, -2]
        UL = Ue[:, :-1] + 0.5 * s[:, :-1]
        UR = Ue[:, 1:] - 0.5 * s[:, 1:]
        return UL, UR

    def rhs(self, U):
        UL, UR = self._face_states(U)
        F = numerical_flux(UL, UR, self.gas, self.flux_kind)
        dU = -(F[:, 1:] - F[:, :-1]) / self.mesh.dx
        speed = max(self.gas.max_speed(UL), self.gas.max_speed(UR))
        return dU, speed

    def run(self):
        U = self.U.copy()
        t_end = self.cfg["time"]["t_end"]
        cfl = self.cfg["time"]["cfl"]
        dt = 0.0
        t = 0.0
        steps = 0
        while t < t_end:
            dU, speed = self.rhs(U)
            if not np.isfinite(speed) or speed <= 0.0:
                raise RuntimeError("non-positive or non-finite maximum wave speed")
            dt = cfl * self.mesh.dx / speed
            if t + dt >= t_end:
                dt = t_end - t
            if self.order == 1:
                U = U + dt * dU
            else:
                U1 = U + dt * dU
                dU1, _ = self.rhs(U1)
                U = 0.5 * U + 0.5 * (U1 + dt * dU1)
            t += dt
            steps += 1
        self.U = U
        rho, u, p = self.gas.primitive(U)
        return self._result(rho, u, p, dt, steps)

    def _result(self, rho, u, p, dt, steps):
        try:
            rho_e, u_e, p_e = exact_fields(
                self.cfg, self.mesh.centers, self.cfg["time"]["t_end"]
            )
        except Exception as exc:  # exact solution is optional
            rho_e = u_e = p_e = None
            exact_error = str(exc)
        else:
            exact_error = None
        return {
            "x": self.mesh.centers,
            "rho": rho,
            "u": u,
            "p": p,
            "rho_exact": rho_e,
            "u_exact": u_e,
            "p_exact": p_e,
            "exact_error": exact_error,
            "dt": dt,
            "steps": steps,
        }


class DGSolver:
    """Discontinuous-Galerkin solver with P0/P1/P2 Legendre data."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.mesh = _Mesh(cfg)
        self.gas = StiffenedGas(cfg["fluids"]["gamma"], cfg["fluids"]["p_inf"])
        self.order = cfg["order"]
        self.M = self.order + 1
        self.flux_kind = cfg["flux"]
        self.tvb = cfg["limiter"]["tvb"]
        self.W = max(4, self.M + 1)
        self.xq_ref, self.wq = roots_legendre(self.W)
        self.wq = 0.5 * self.wq
        self.U = self._initial_data()

    # ------------------------------------------------------------------
    # basis functions
    # ------------------------------------------------------------------
    def basis(self, x, xc, dx):
        x_arr = np.asarray(x, dtype=float)
        xi = (x_arr - xc) / (0.5 * dx)
        B = np.zeros((self.M,) + xi.shape, dtype=float)
        B[0] = 1.0
        if self.M > 1:
            B[1] = xi
        if self.M > 2:
            B[2] = xi * xi - 1.0 / 3.0
        return B

    def basis_x(self, x, xc, dx):
        x_arr = np.asarray(x, dtype=float)
        B = np.zeros((self.M,) + x_arr.shape, dtype=float)
        if self.M > 1:
            B[1] = 1.0 / (0.5 * dx)
        if self.M > 2:
            B[2] = 2.0 * (x_arr - xc) / ((0.5 * dx) ** 2)
        return B

    # ------------------------------------------------------------------
    def _initial_data(self):
        U = np.zeros((3, self.M, self.mesh.nx), dtype=float)
        for i in range(self.mesh.nx):
            xq = 0.5 * self.mesh.dx * self.xq_ref + self.mesh.centers[i]
            rho, u, p = primitive_at(self.cfg, xq)
            Uq = self.gas.conservative(rho, u, p)
            B = self.basis(xq, self.mesh.centers[i], self.mesh.dx)
            Mmat = (B * self.wq) @ B.T * self.mesh.dx
            U[:, :, i] = ((Uq * self.wq) @ B.T * self.mesh.dx) / np.diagonal(Mmat)
        return U

    # ------------------------------------------------------------------
    def _face_states(self, U):
        n = self.mesh.nx
        xl = self.mesh.edges[:-1]
        xr = self.mesh.edges[1:]
        Bl = self.basis(xl, self.mesh.centers, self.mesh.dx)
        Br = self.basis(xr, self.mesh.centers, self.mesh.dx)
        Ul = np.zeros((3, n + 1), dtype=float)
        Ur = np.zeros((3, n + 1), dtype=float)
        # Face j separates cell j-1 (left) and cell j (right).  The left
        # state therefore comes from the *right* edge of cell j-1 and the
        # right state from the *left* edge of cell j.
        Ul[:, 1:] = np.sum(Br * U, axis=1)
        Ur[:, :-1] = np.sum(Bl * U, axis=1)
        bc = self.mesh.boundary
        if bc == "periodic":
            Ul[:, 0] = Ul[:, -1]
            Ur[:, -1] = Ur[:, 0]
        elif bc == "transmissive":
            Ul[:, 0] = Ur[:, 0]
            Ur[:, -1] = Ul[:, -1]
        elif bc == "reflective":
            Ul[:, 0] = _reflect(Ur[:, 0])
            Ur[:, -1] = _reflect(Ul[:, -1])
        else:  # pragma: no cover
            raise ValueError(bc)
        return Ul, Ur

    def rhs(self, U):
        n = self.mesh.nx
        Ul, Ur = self._face_states(U)
        F = numerical_flux(Ul, Ur, self.gas, self.flux_kind)
        dU = np.zeros_like(U)
        for i in range(n):
            center = self.mesh.centers[i]
            dx = self.mesh.dx
            B_lr = self.basis([self.mesh.edges[i], self.mesh.edges[i + 1]], center, dx)
            V1 = np.outer(F[:, i + 1], B_lr[:, 1]) - np.outer(F[:, i], B_lr[:, 0])
            xq = 0.5 * dx * self.xq_ref + center
            Bq = self.basis(xq, center, dx)
            Bxq = self.basis_x(xq, center, dx)
            Uq = U[:, :, i] @ Bq
            Fq = self.gas.flux(Uq)
            V2 = -(dx * (Fq @ (Bxq * self.wq).T))
            Mmat = (Bq * self.wq) @ Bq.T * dx
            # The reference DG code stores the negative time derivative and
            # updates with ``U - dt*dU``.  Keep the public RHS convention
            # ``dU/dt`` instead, so both FV and DG can use ``U + dt*dU``.
            dU[:, :, i] = -(V1 + V2) / np.diagonal(Mmat)
        speed = max(self.gas.max_speed(Ul), self.gas.max_speed(Ur))
        return dU, speed

    # ------------------------------------------------------------------
    # limiters
    # ------------------------------------------------------------------
    def _differences(self):
        U = self.U
        n = self.mesh.nx
        DU_p = np.zeros((3, n), dtype=float)
        DU_m = np.zeros((3, n), dtype=float)
        DU_p[:, :-1] = U[:, 0, 1:] - U[:, 0, :-1]
        DU_m[:, 1:] = U[:, 0, 1:] - U[:, 0, :-1]
        if self.mesh.boundary == "periodic":
            DU_p[:, -1] = U[:, 0, 0] - U[:, 0, -1]
            DU_m[:, 0] = U[:, 0, 0] - U[:, 0, -1]
        return DU_p, DU_m

    def limit(self, U):
        self.U = U
        if self.M == 1:
            return U
        DU_p, DU_m = self._differences()
        if self.M == 2:
            Br = self.basis(self.mesh.edges[1:], self.mesh.centers, self.mesh.dx)
            U_r = np.sum(Br * U, axis=1)
            dUp = U_r - U[:, 0, :]
            dUp_mod = minmod(dUp, DU_p, DU_m, self.tvb, self.mesh.dx)
            U[:, 1, :] = dUp_mod
            return U

        # P2 characteristic TVB limiter (same construction as the reference
        # DG code, written with the class basis).
        Bl = self.basis(self.mesh.edges[:-1], self.mesh.centers, self.mesh.dx)
        Br = self.basis(self.mesh.edges[1:], self.mesh.centers, self.mesh.dx)
        U_l = np.sum(Bl * U, axis=1)
        U_r = np.sum(Br * U, axis=1)
        U0 = U[:, 0, :]
        dUr = U_r - U0
        dUl = U0 - U_l
        eps = 1.0e-12
        for i in range(self.mesh.nx):
            gas = self.gas
            rho, u, p = gas.primitive(U0[:, i])
            c = gas.sound_speed(U0[:, i])
            H = (U0[2, i] + p) / rho
            R = np.array([[1.0, 1.0, 1.0],
                          [u - c, u, u + c],
                          [H - u * c, 0.5 * u * u, H + u * c]]) + eps * np.eye(3)
            R_inv = np.linalg.inv(R)
            dUr_mod = R @ minmod(R_inv @ dUr[:, i], R_inv @ DU_p[:, i],
                                 R_inv @ DU_m[:, i], self.tvb, self.mesh.dx)
            dUl_mod = R @ minmod(R_inv @ dUl[:, i], R_inv @ DU_p[:, i],
                                 R_inv @ DU_m[:, i], self.tvb, self.mesh.dx)
            U[:, 1, i] = 0.5 * (dUr_mod + dUl_mod)
            U[:, 2, i] = 0.75 * (dUr_mod - dUl_mod)
        return U

    # ------------------------------------------------------------------
    def run(self):
        U = self.U.copy()
        t_end = self.cfg["time"]["t_end"]
        cfl = self.cfg["time"]["cfl"]
        t = 0.0
        dt = 0.0
        steps = 0
        while t < t_end:
            dU, speed = self.rhs(U)
            if not np.isfinite(speed) or speed <= 0.0:
                raise RuntimeError("non-positive or non-finite maximum wave speed")
            dt = cfl * self.mesh.dx / speed
            if t + dt >= t_end:
                dt = t_end - t
            if self.M == 1:
                U = U + dt * dU
            elif self.M == 2:
                U_cp = U.copy()
                U = self.limit(U + dt * dU)
                dU1, _ = self.rhs(U)
                U = self.limit(0.5 * U_cp + 0.5 * (U + dt * dU1))
            else:
                U_cp = U.copy()
                U = self.limit(U + dt * dU)
                dU1, _ = self.rhs(U)
                U = self.limit(0.75 * U_cp + 0.25 * (U + dt * dU1))
                dU2, _ = self.rhs(U)
                U = self.limit((1.0 / 3.0) * U_cp + (2.0 / 3.0) * (U + dt * dU2))
            t += dt
            steps += 1
        self.U = U
        U0 = U[:, 0, :]
        rho, u, p = self.gas.primitive(U0)
        return self._result(rho, u, p, dt, steps)

    def _result(self, rho, u, p, dt, steps):
        try:
            rho_e, u_e, p_e = exact_fields(
                self.cfg, self.mesh.centers, self.cfg["time"]["t_end"]
            )
        except Exception as exc:
            rho_e = u_e = p_e = None
            exact_error = str(exc)
        else:
            exact_error = None
        return {
            "x": self.mesh.centers,
            "rho": rho,
            "u": u,
            "p": p,
            "rho_exact": rho_e,
            "u_exact": u_e,
            "p_exact": p_e,
            "exact_error": exact_error,
            "dt": dt,
            "steps": steps,
        }


def solve_single(cfg: dict) -> dict:
    if cfg["scheme"] == "fv":
        return FVSolver(cfg).run()
    if cfg["scheme"] == "dg":
        return DGSolver(cfg).run()
    raise ValueError(cfg["scheme"])
