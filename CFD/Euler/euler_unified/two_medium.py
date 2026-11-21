"""Sharp-interface two-medium Euler solver on a fixed Eulerian grid.

The implementation follows the cut-cell DG algorithm used in the original
``DGEuler2mediumST_P1/P2.py`` scripts.  The partial cell containing the
material interface is represented by two independent phase elements; the
interface is advanced with the exact Riemann solver.  With ``order=0`` the
same code is a first-order finite-volume cut-cell scheme.
"""

from __future__ import annotations

import numpy as np
from scipy.special import roots_legendre

from .eos import StiffenedGas
from .initial import exact_fields
from .numerics import minmod, numerical_flux


# The exact Riemann modules are imported lazily inside _interface_velocity.
_FDEXACT = None
_ERS = None


def _load_riemann():
    global _FDEXACT, _ERS
    if _FDEXACT is None:
        import sys
        from pathlib import Path

        path = Path(__file__).resolve().parents[2] / "Riemann" / "code"
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
        import Fluid as Fdexact
        import Exact_Riemann_solver as ERS
        _FDEXACT = Fdexact
        _ERS = ERS
    return _FDEXACT, _ERS


class TwoMediumSolver:
    def __init__(self, cfg):
        self.cfg = cfg
        mesh = cfg["mesh"]
        self.x_min = mesh["x_min"]
        self.x_max = mesh["x_max"]
        self.nx = mesh["nx"]
        self.dx = (self.x_max - self.x_min) / self.nx
        self.point = np.linspace(self.x_min, self.x_max, self.nx + 1)
        self.n_edge = self.nx + 1
        self.ele_vol = self.dx
        self.ele_centroid = 0.5 * (self.point[:-1] + self.point[1:])
        self.boundary = mesh["boundary"]

        # A two-medium finite-volume case is the P0 version of the same
        # cut-cell algorithm.
        self.order = 0 if cfg["scheme"] == "fv" else cfg["order"]
        self.M = self.order + 1
        self.flux_kind = cfg["flux"]
        self.tvb = cfg["limiter"]["tvb"]
        self.eps = 1.0e-12

        self.gas0 = StiffenedGas(cfg["fluids"]["left"]["gamma"],
                                 cfg["fluids"]["left"]["p_inf"])
        self.gas1 = StiffenedGas(cfg["fluids"]["right"]["gamma"],
                                 cfg["fluids"]["right"]["p_inf"])

        self.W = max(4, self.M + 1)
        self.xq_ref, self.wq = roots_legendre(self.W)
        self.wq = 0.5 * self.wq

        # Interface geometry.
        self.x_I = self._safe_interface(cfg["initial"]["x_interface"])
        self.x_I_cp = self.x_I
        self.phi = self.point - self.x_I

        self.point_label = np.zeros(self.n_edge, dtype=int)
        self.point_label_new = np.zeros(self.n_edge, dtype=int)
        self.ele_label = np.zeros(self.nx, dtype=int)
        self.ele_label_new = np.zeros(self.nx, dtype=int)
        self.idx = -1
        self.idx_new = -1

        self.phase_vol = np.zeros((2, self.nx))
        self.phase_centroid = np.zeros((2, self.nx))
        self.phase_vol_new = np.zeros((2, self.nx))
        self.phase_centroid_new = np.zeros((2, self.nx))
        self.phase_vol_cp = np.zeros((2, self.nx))
        self.phase_centroid_cp = np.zeros((2, self.nx))
        self.IC_vol = np.zeros(2)
        self.IC_centroid = np.zeros(2)
        self.IC_vol_new = np.zeros(2)
        self.IC_centroid_new = np.zeros(2)

        self.U_data = np.zeros((3, self.M, self.nx, 2))
        self.U_IC = np.zeros((3, self.M, 2))
        self.U_IC_cp = np.zeros((3, self.M, 2))
        self.U_data_cp = np.zeros_like(self.U_data)
        self.IC_flux = np.zeros(3)

        self._mark(self.phi)
        self._phase_info(self.x_I)
        self._interface_info(idx=self.idx)
        self._init_data()
        self._compose()

    # ------------------------------------------------------------------
    # basis
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

    def basis_single(self, x, xc, dx):
        return self.basis(np.asarray([x], dtype=float), xc, dx)[:, 0]

    # ------------------------------------------------------------------
    # geometry / marking
    # ------------------------------------------------------------------
    def _safe_interface(self, x_i):
        """Avoid placing the interface exactly on a grid point.

        A tiny shift keeps two positive phase volumes in the cut cell; the
        shift is far below the discretisation error.
        """
        x_i = float(x_i)
        for edge in self.point[1:-1]:
            if abs(edge - x_i) < 1.0e-12 * self.dx:
                return edge + 1.0e-8 * self.dx
        return x_i

    def _mark(self, phi, new=False):
        pl = (np.asarray(phi) > 0.0).astype(int)
        el = pl[:-1] + pl[1:]
        idx = np.where(el == 1)[0]
        if len(idx) == 0:
            raise RuntimeError("the material interface is not inside the domain")
        if new:
            self.point_label_new[:] = pl
            self.ele_label_new[:] = el
            self.idx_new = int(idx[0])
        else:
            self.point_label[:] = pl
            self.ele_label[:] = el
            self.idx = int(idx[0])

    def _phase_info(self, x_i, new=False):
        vol = self.phase_vol_new if new else self.phase_vol
        cen = self.phase_centroid_new if new else self.phase_centroid
        lab = self.ele_label_new if new else self.ele_label
        cen[0, :] = self.ele_centroid
        cen[1, :] = self.ele_centroid
        vol[0, :] = self.ele_vol * (lab < 1)
        vol[1, :] = self.ele_vol * (lab > 1)
        for i in range(self.nx):
            if lab[i] == 1:
                vol[0, i] = abs(x_i - self.point[i])
                vol[1, i] = abs(self.point[i + 1] - x_i)
                cen[0, i] = 0.5 * (x_i + self.point[i])
                cen[1, i] = 0.5 * (self.point[i + 1] + x_i)

    def _interface_info(self, new=False, idx=None):
        vol = self.phase_vol_new if new else self.phase_vol
        cen = self.phase_centroid_new if new else self.phase_centroid
        if idx is None:
            idx = self.idx_new if new else self.idx
        sl = slice(idx - 1, idx + 2)
        v = np.sum(vol[:, sl], axis=1)
        c = np.sum(cen[:, sl] * vol[:, sl], axis=1) / v
        if new:
            self.IC_vol_new = v
            self.IC_centroid_new = c
        else:
            self.IC_vol = v
            self.IC_centroid = c

    # ------------------------------------------------------------------
    # initial data
    # ------------------------------------------------------------------
    def _init_data(self):
        initial = self.cfg["initial"]
        left = initial["left"]
        right = initial["right"]
        for i in range(self.nx):
            for phase in (0, 1):
                vol = self.phase_vol[phase, i]
                if vol <= self.eps:
                    self.U_data[:, :, i, phase] = 0.0
                    continue
                if initial["type"] == "smooth":
                    cen = self.phase_centroid[phase, i]
                    xq = 0.5 * vol * self.xq_ref + cen
                    rho = initial["rho0"] + initial["rho_amp"] * np.sin(initial["k"] * np.pi * xq)
                    u = np.full_like(xq, initial["u0"])
                    p = np.full_like(xq, initial["p0"])
                    gas = self.gas0 if phase == 0 else self.gas1
                    Uq = gas.conservative(rho, u, p)
                    B = self.basis(xq, cen, vol)
                    Mmat = (B * self.wq) @ B.T * vol
                    self.U_data[:, :, i, phase] = (
                        (Uq * self.wq) @ B.T * vol
                    ) / np.diagonal(Mmat)
                else:
                    state = left if phase == 0 else right
                    gas = self.gas0 if phase == 0 else self.gas1
                    Uc = gas.conservative(state["rho"], state["u"], state["p"])
                    self.U_data[:, :, i, phase] = 0.0
                    self.U_data[:, 0, i, phase] = Uc

    # ------------------------------------------------------------------
    # interface element transfer
    # ------------------------------------------------------------------
    def _compose(self):
        for phase in (0, 1):
            Uf = np.zeros((3, self.M, 3))
            for j, i in enumerate(range(self.idx - 1, self.idx + 2)):
                vol = self.phase_vol[phase, i]
                if vol > self.eps:
                    cen = self.phase_centroid[phase, i]
                    xq = 0.5 * vol * self.xq_ref + cen
                    Bs = self.basis(xq, cen, vol)
                    Bi = self.basis(xq, self.IC_centroid[phase], self.IC_vol[phase])
                    Uf[:, :, j] = (self.U_data[:, :, i, phase] @ Bs) @ (Bi * self.wq).T * vol
            xq = 0.5 * self.IC_vol[phase] * self.xq_ref + self.IC_centroid[phase]
            Bi = self.basis(xq, self.IC_centroid[phase], self.IC_vol[phase])
            Mmat = (Bi * self.wq) @ Bi.T * self.IC_vol[phase]
            self.U_IC[:, :, phase] = np.sum(Uf, axis=2) / np.diagonal(Mmat)

    def _decompose(self, idx):
        for phase in (0, 1):
            for i in range(idx - 1, idx + 2):
                vol = self.phase_vol_new[phase, i]
                if vol > self.eps:
                    cen = self.phase_centroid_new[phase, i]
                    xq = 0.5 * vol * self.xq_ref + cen
                    Bs = self.basis(xq, cen, vol)
                    Bi = self.basis(xq, self.IC_centroid_new[phase],
                                    self.IC_vol_new[phase])
                    F = (self.U_IC[:, :, phase] @ Bi) @ (Bs * self.wq).T * vol
                    Mi = (Bs * self.wq) @ Bs.T * vol
                    self.U_data[:, :, i, phase] = F / np.diagonal(Mi)
                else:
                    self.U_data[:, :, i, phase] = 0.0

    def _uic_old_mass(self):
        out = np.zeros(2)
        for phase in (0, 1):
            xq = 0.5 * self.IC_vol[phase] * self.xq_ref + self.IC_centroid[phase]
            B = self.basis(xq, self.IC_centroid[phase], self.IC_vol[phase])
            M = (B * self.wq) @ B.T * self.IC_vol[phase]
            out[phase] = np.diagonal(M)[0]
        return out

    def _uic_new_mass(self):
        out = np.zeros(2)
        for phase in (0, 1):
            xq = 0.5 * self.IC_vol_new[phase] * self.xq_ref + self.IC_centroid_new[phase]
            B = self.basis(xq, self.IC_centroid_new[phase], self.IC_vol_new[phase])
            M = (B * self.wq) @ B.T * self.IC_vol_new[phase]
            out[phase] = np.diagonal(M)[0]
        return out

    def _update_uic(self, dt, dUIC, stage):
        """Advance the interface-element coefficients.

        ``dUIC`` is the time derivative of the *integrated* moment,
        ``d/dt int U B dx``, so the mass matrices of the old and new
        interface elements appear explicitly.
        """
        M_old = self._uic_old_mass()
        M_new = self._uic_new_mass()
        for phase in (0, 1):
            old = self.U_IC[:, :, phase] * M_old[phase]
            if stage == 1:
                value = old + dt * dUIC[:, :, phase]
            elif stage == 2:
                value = 0.5 * self.U_IC_cp[:, :, phase] * M_old[phase] + 0.5 * (
                    old + dt * dUIC[:, :, phase])
            elif stage == "3_2":
                value = (0.75 * self.U_IC_cp[:, :, phase] * M_old[phase]
                         + 0.25 * (old + dt * dUIC[:, :, phase]))
            elif stage == "3_3":
                value = ((1.0 / 3.0) * self.U_IC_cp[:, :, phase] * M_old[phase]
                         + (2.0 / 3.0) * (old + dt * dUIC[:, :, phase]))
            else:
                raise ValueError(stage)
            self.U_IC[:, :, phase] = value / M_new[phase]

    def _backup_uic_cp(self, idx_new):
        # Geometry of the old (beginning-of-step) state.
        vol_cp = self.phase_vol_cp[:, idx_new - 1:idx_new + 2]
        cen_cp = self.phase_centroid_cp[:, idx_new - 1:idx_new + 2]
        temp_v = np.sum(vol_cp, axis=1)
        temp_c = np.sum(cen_cp * vol_cp, axis=1) / temp_v
        temp_v_new = np.sum(self.phase_vol_new[:, idx_new - 1:idx_new + 2], axis=1)
        temp_c_new = np.sum(
            self.phase_centroid_new[:, idx_new - 1:idx_new + 2]
            * self.phase_vol_new[:, idx_new - 1:idx_new + 2], axis=1
        ) / temp_v_new
        for phase in (0, 1):
            Uf = np.zeros((3, self.M, 3))
            for j, i in enumerate(range(idx_new - 1, idx_new + 2)):
                if self.phase_vol_cp[phase, i] > self.eps:
                    vol = self.phase_vol_cp[phase, i]
                    cen = self.phase_centroid_cp[phase, i]
                    xq = 0.5 * vol * self.xq_ref + cen
                    Bs = self.basis(xq, cen, vol)
                    Bi = self.basis(xq, temp_c[phase], temp_v[phase])
                    Uf[:, :, j] = (self.U_data_cp[:, :, i, phase] @ Bs) @ (Bi * self.wq).T * vol
            xq = 0.5 * temp_v[phase] * self.xq_ref + temp_c[phase]
            Bi = self.basis(xq, temp_c[phase], temp_v[phase])
            M_temp = (Bi * self.wq) @ Bi.T * temp_v[phase]
            uic = np.sum(Uf, axis=2) / np.diagonal(M_temp)
            # Rescale the polynomial to the new interface element.
            xq_new = 0.5 * temp_v_new[phase] * self.xq_ref + temp_c_new[phase]
            Bi_new = self.basis(xq_new, temp_c_new[phase], temp_v_new[phase])
            M_new = (Bi_new * self.wq) @ Bi_new.T * temp_v_new[phase]
            self.U_IC_cp[:, :, phase] = uic * np.diagonal(M_temp) / np.diagonal(M_new)

    # ------------------------------------------------------------------
    # exact interface Riemann problem
    # ------------------------------------------------------------------
    def _interface_velocity(self):
        rho0, u0, p0 = self.gas0.primitive(self.U_IC[:, 0, 0])
        rho1, u1, p1 = self.gas1.primitive(self.U_IC[:, 0, 1])
        Fdexact, ERS = _load_riemann()
        WL = Fdexact.Fluid(float(rho0), float(u0), float(p0),
                           self.gas0.gamma, self.gas0.p_inf)
        WR = Fdexact.Fluid(float(rho1), float(u1), float(p1),
                           self.gas1.gamma, self.gas1.p_inf)
        try:
            p_star, u_star = ERS.ExactRiemannSolver(WL, WR, 1.0e-12, 1000).solver()
        except Exception:
            p_star = 0.5 * (p0 + p1)
            u_star = 0.5 * (u0 + u1)
        self.IC_flux[0] = 0.0
        self.IC_flux[1] = p_star
        self.IC_flux[2] = p_star * u_star
        return float(u_star)

    # ------------------------------------------------------------------
    # spatial residual
    # ------------------------------------------------------------------
    def rhs(self):
        n = self.nx
        idx = self.idx
        U_data = self.U_data
        U_IC = self.U_IC
        dU = np.zeros_like(U_data)
        dUIC = np.zeros((3, self.M, 2))

        edge_l = self.basis(self.point[1:], self.ele_centroid, self.ele_vol)
        edge_r = self.basis(self.point[:-1], self.ele_centroid, self.ele_vol)
        Ul = np.zeros((3, self.n_edge))
        Ur = np.zeros((3, self.n_edge))

        # Phase 0 cells strictly to the left of the interface cell.
        if idx - 1 > 0:
            Ul[:, 1:idx] = np.sum(
                edge_l[:, :idx - 1] * U_data[:, :, :idx - 1, 0], axis=1)
            Ur[:, :idx - 1] = np.sum(
                edge_r[:, :idx - 1] * U_data[:, :, :idx - 1, 0], axis=1)
        # Phase 1 cells strictly to the right of the interface cell.
        if idx + 2 < n:
            Ul[:, idx + 3:] = np.sum(
                edge_l[:, idx + 2:] * U_data[:, :, idx + 2:, 1], axis=1)
            Ur[:, idx + 2:-1] = np.sum(
                edge_r[:, idx + 2:] * U_data[:, :, idx + 2:, 1], axis=1)

        # Outer edges of the interface element.
        Br0 = self.basis_single(self.point[idx - 1], self.IC_centroid[0],
                                self.IC_vol[0])
        Bl1 = self.basis_single(self.point[idx + 2], self.IC_centroid[1],
                                self.IC_vol[1])
        Ur[:, idx - 1] = np.sum(Br0 * U_IC[:, :, 0], axis=1)
        Ul[:, idx + 2] = np.sum(Bl1 * U_IC[:, :, 1], axis=1)

        # Domain boundaries.
        if self.boundary == "periodic":
            Ul[:, 0] = Ul[:, -1]
            Ur[:, -1] = Ur[:, 0]
        else:  # transmissive
            Ul[:, 0] = Ur[:, 0]
            Ur[:, -1] = Ul[:, -1]

        Fu = np.zeros((3, self.n_edge))
        max_speed = 0.0
        for i in range(self.n_edge):
            if idx <= i <= idx + 1:
                continue
            gas = self.gas0 if i <= idx - 1 else self.gas1
            F = numerical_flux(Ul[:, i:i + 1], Ur[:, i:i + 1], gas,
                               self.flux_kind)[:, 0]
            Fu[:, i] = F
            max_speed = max(max_speed,
                            float(gas.max_speed(Ul[:, i:i + 1])),
                            float(gas.max_speed(Ur[:, i:i + 1])))

        # Regular cells.
        for i in range(n):
            if idx - 1 <= i <= idx + 1:
                continue
            phase = 0 if i < idx - 1 else 1
            gas = self.gas0 if phase == 0 else self.gas1
            Blr = self.basis([self.point[i], self.point[i + 1]],
                             self.ele_centroid[i], self.ele_vol)
            V1 = (np.outer(Fu[:, i + 1], Blr[:, 1])
                  - np.outer(Fu[:, i], Blr[:, 0]))
            xq = 0.5 * self.ele_vol * self.xq_ref + self.ele_centroid[i]
            Bq = self.basis(xq, self.ele_centroid[i], self.ele_vol)
            Bxq = self.basis_x(xq, self.ele_centroid[i], self.ele_vol)
            Uq = U_data[:, :, i, phase] @ Bq
            Fq = gas.flux(Uq)
            V2 = -(self.ele_vol * (Fq @ (Bxq * self.wq).T))
            Mmat = (Bq * self.wq) @ Bq.T * self.ele_vol
            # dU/dt = -(V1 + V2) / M  (V1/V2 use the DG boundary sign).
            dU[:, :, i, phase] = -(V1 + V2) / np.diagonal(Mmat)

        # Interface element, one equation per phase.
        edge_idx = (idx - 1, idx + 2)
        for phase in (0, 1):
            gas = self.gas0 if phase == 0 else self.gas1
            vol = self.IC_vol[phase]
            cen = self.IC_centroid[phase]
            Blr = self.basis([cen - 0.5 * vol, cen + 0.5 * vol], cen, vol)
            if phase == 0:
                V1 = (np.outer(self.IC_flux, Blr[:, 1])
                      - np.outer(Fu[:, edge_idx[0]], Blr[:, 0]))
            else:
                V1 = (np.outer(Fu[:, edge_idx[1]], Blr[:, 1])
                      - np.outer(self.IC_flux, Blr[:, 0]))
            xq = 0.5 * vol * self.xq_ref + cen
            Bq = self.basis(xq, cen, vol)
            Bxq = self.basis_x(xq, cen, vol)
            Uq = U_IC[:, :, phase] @ Bq
            Fq = gas.flux(Uq)
            V2 = -(vol * (Fq @ (Bxq * self.wq).T))
            dUIC[:, :, phase] = -(V1 + V2)
        return dU, dUIC, max_speed

    # ------------------------------------------------------------------
    # limiter
    # ------------------------------------------------------------------
    def limit(self):
        if self.M == 1:
            return
        n = self.nx
        idx = self.idx
        U = self.U_data
        U_IC = self.U_IC
        DU_p = np.zeros((3, n, 2))
        DU_m = np.zeros((3, n, 2))
        DU_p[:, :-1, :] = U[:, 0, 1:, :] - U[:, 0, :-1, :]
        DU_m[:, 1:, :] = U[:, 0, 1:, :] - U[:, 0, :-1, :]
        DU_p[:, idx - 2, 0] = U_IC[:, 0, 0] - U[:, 0, idx - 2, 0]
        DU_m[:, idx + 2, 1] = U[:, 0, idx + 2, 1] - U_IC[:, 0, 1]

        if self.M == 2:
            Br = self.basis(self.point[1:], self.ele_centroid, self.ele_vol)
            U_r0 = np.sum(Br * U[:, :, :, 0], axis=1)
            U_r1 = np.sum(Br * U[:, :, :, 1], axis=1)
            dUp0 = U_r0 - U[:, 0, :, 0]
            dUp1 = U_r1 - U[:, 0, :, 1]
            dUp_mod0 = minmod(dUp0, DU_p[:, :, 0], DU_m[:, :, 0],
                              self.tvb, self.ele_vol)
            dUp_mod1 = minmod(dUp1, DU_p[:, :, 1], DU_m[:, :, 1],
                              self.tvb, self.ele_vol)
            U[:, 1, :idx - 1, 0] = dUp_mod0[:, :idx - 1]
            U[:, 1, idx + 2:, 1] = dUp_mod1[:, idx + 2:]

            U0IC0 = U_IC[:, 0, 0]
            U0IC1 = U_IC[:, 0, 1]
            Bl0 = self.basis_single(self.point[idx - 1], self.IC_centroid_new[0],
                                    self.IC_vol_new[0])
            Br1 = self.basis_single(self.point[idx + 2], self.IC_centroid_new[1],
                                    self.IC_vol_new[1])
            U_IC0_l = np.sum(Bl0 * U_IC[:, :, 0], axis=1)
            U_IC1_r = np.sum(Br1 * U_IC[:, :, 1], axis=1)
            dUp_IC0 = U0IC0 - U_IC0_l
            dUp_IC1 = U_IC1_r - U0IC1
            dUp_IC0_mod = minmod(dUp_IC0, DU_p[:, idx - 2, 0],
                                 U_IC[:, 0, 1] - U_IC[:, 0, 0],
                                 self.tvb, self.IC_vol_new[0])
            dUp_IC1_mod = minmod(dUp_IC1, U_IC[:, 0, 1] - U_IC[:, 0, 0],
                                 DU_m[:, idx + 2, 1],
                                 self.tvb, self.IC_vol_new[1])
            U_IC[:, 1, 0] = dUp_IC0_mod
            U_IC[:, 1, 1] = dUp_IC1_mod
            return

        # P2 characteristic TVB limiter.
        Bl = self.basis(self.point[:-1], self.ele_centroid, self.ele_vol)
        Br = self.basis(self.point[1:], self.ele_centroid, self.ele_vol)
        U_l0 = np.sum(Bl * U[:, :, :, 0], axis=1)
        U_l1 = np.sum(Bl * U[:, :, :, 1], axis=1)
        U_r0 = np.sum(Br * U[:, :, :, 0], axis=1)
        U_r1 = np.sum(Br * U[:, :, :, 1], axis=1)
        U0_0 = U[:, 0, :, 0]
        U0_1 = U[:, 0, :, 1]
        dUr0 = U_r0 - U0_0
        dUr1 = U_r1 - U0_1
        dUl0 = U0_0 - U_l0
        dUl1 = U0_1 - U_l1

        def char_matrix(uc, gas):
            rho, u, p = gas.primitive(uc)
            c = gas.sound_speed(uc)
            H = (uc[2] + p) / rho
            return np.array([[1.0, 1.0, 1.0],
                             [u - c, u, u + c],
                             [H - u * c, 0.5 * u * u, H + u * c]]) + 1.0e-12 * np.eye(3)

        for i in range(idx - 1):
            R = char_matrix(U0_0[:, i], self.gas0)
            R_inv = np.linalg.inv(R)
            dUr_mod = R @ minmod(R_inv @ dUr0[:, i], R_inv @ DU_p[:, i, 0],
                                 R_inv @ DU_m[:, i, 0], self.tvb, self.ele_vol)
            dUl_mod = R @ minmod(R_inv @ dUl0[:, i], R_inv @ DU_p[:, i, 0],
                                 R_inv @ DU_m[:, i, 0], self.tvb, self.ele_vol)
            U[:, 1, i, 0] = 0.5 * (dUr_mod + dUl_mod)
            U[:, 2, i, 0] = 0.75 * (dUr_mod - dUl_mod)
        for i in range(idx + 2, n):
            R = char_matrix(U0_1[:, i], self.gas1)
            R_inv = np.linalg.inv(R)
            dUr_mod = R @ minmod(R_inv @ dUr1[:, i], R_inv @ DU_p[:, i, 1],
                                 R_inv @ DU_m[:, i, 1], self.tvb, self.ele_vol)
            dUl_mod = R @ minmod(R_inv @ dUl1[:, i], R_inv @ DU_p[:, i, 1],
                                 R_inv @ DU_m[:, i, 1], self.tvb, self.ele_vol)
            U[:, 1, i, 1] = 0.5 * (dUr_mod + dUl_mod)
            U[:, 2, i, 1] = 0.75 * (dUr_mod - dUl_mod)

        # Interface element.
        U0IC0 = U_IC[:, 0, 0]
        U0IC1 = U_IC[:, 0, 1]
        Bl0 = self.basis_single(self.point[idx - 1], self.IC_centroid_new[0],
                                self.IC_vol_new[0])
        Br0 = self.basis_single(self.x_I, self.IC_centroid_new[0],
                                self.IC_vol_new[0])
        Bl1 = self.basis_single(self.x_I, self.IC_centroid_new[1],
                                self.IC_vol_new[1])
        Br1 = self.basis_single(self.point[idx + 2], self.IC_centroid_new[1],
                                self.IC_vol_new[1])
        U_IC0_l = np.sum(Bl0 * U_IC[:, :, 0], axis=1)
        U_IC0_r = np.sum(Br0 * U_IC[:, :, 0], axis=1)
        U_IC1_l = np.sum(Bl1 * U_IC[:, :, 1], axis=1)
        U_IC1_r = np.sum(Br1 * U_IC[:, :, 1], axis=1)
        dUr0 = U_IC0_r - U0IC0
        dUl0 = U0IC0 - U_IC0_l
        dUr1 = U_IC1_r - U0IC1
        dUl1 = U0IC1 - U_IC1_l

        for phase, (dUrp, dUlp, U0p, gas, vol_new) in enumerate((
            (dUr0, dUl0, U0IC0, self.gas0, self.IC_vol_new[0]),
            (dUr1, dUl1, U0IC1, self.gas1, self.IC_vol_new[1]),
        )):
            R = char_matrix(U0p, gas)
            R_inv = np.linalg.inv(R)
            if phase == 0:
                b = DU_p[:, idx - 2, 0]
                c = U0IC1 - U0IC0
            else:
                b = U0IC1 - U0IC0
                c = DU_m[:, idx + 2, 1]
            dUr_mod = R @ minmod(R_inv @ dUrp, R_inv @ b, R_inv @ c,
                                 self.tvb, vol_new)
            dUl_mod = R @ minmod(R_inv @ dUlp, R_inv @ b, R_inv @ c,
                                 self.tvb, vol_new)
            U_IC[:, 1, phase] = 0.5 * (dUr_mod + dUl_mod)
            U_IC[:, 2, phase] = 0.75 * (dUr_mod - dUl_mod)

    # ------------------------------------------------------------------
    # driver
    # ------------------------------------------------------------------
    def run(self):
        t_end = self.cfg["time"]["t_end"]
        cfl = self.cfg["time"]["cfl"]
        t = 0.0
        dt = 0.0
        steps = 0
        while t < t_end:
            U_data_cp = self.U_data.copy()
            U_IC_cp = self.U_IC.copy()
            x_I_cp = self.x_I
            self.U_data_cp = U_data_cp
            self.U_IC_cp = U_IC_cp
            self.x_I_cp = x_I_cp
            self.phase_vol_cp = self.phase_vol.copy()
            self.phase_centroid_cp = self.phase_centroid.copy()

            # ---- stage 1 -------------------------------------------------
            self._compose()
            u_star = self._interface_velocity()
            dU, dUIC, speed = self.rhs()
            speed = max(speed, abs(u_star))
            if not np.isfinite(speed) or speed <= 0.0:
                raise RuntimeError("non-positive or non-finite maximum wave speed")
            dt = cfl * self.ele_vol / speed
            if t + dt >= t_end:
                dt = t_end - t
            self.x_I_new = self.x_I + u_star * dt
            self.phi = self.point - self.x_I_new
            self._mark(self.phi, new=True)
            self._phase_info(self.x_I_new, new=True)
            # Keep the old interface-cell stencil for the stage update and
            # decomposition; the stencil is recentred at the beginning of
            # the next stage.  This is the convention of the reference
            # cut-cell DG algorithm and avoids losing the cell that leaves
            # the stencil when the interface crosses a grid point.
            self._interface_info(new=True, idx=self.idx)
            self.U_data = self.U_data + dt * dU
            self._update_uic(dt, dUIC, stage=1)
            self.limit()
            self._decompose(self.idx)
            if self.M > 1:
                self._backup_uic_cp(self.idx_new)
            self.x_I = self.x_I_new
            self.idx = self.idx_new
            self._mark(self.point - self.x_I)
            self._phase_info(self.x_I)
            self._interface_info(idx=self.idx)

            # ---- stage 2 -------------------------------------------------
            if self.M > 1:
                self._compose()
                u_star = self._interface_velocity()
                dU, dUIC, _ = self.rhs()
                # SSP-RK2 for P1, first SSP-RK3 substep for P2.
                if self.M == 2:
                    coef_cp, stage = 0.5, 2
                else:
                    coef_cp, stage = 0.75, "3_2"
                self.x_I_new = coef_cp * x_I_cp + (1.0 - coef_cp) * (
                    self.x_I + u_star * dt)
                self.phi = self.point - self.x_I_new
                self._mark(self.phi, new=True)
                self._phase_info(self.x_I_new, new=True)
                self._interface_info(new=True, idx=self.idx)
                self.U_data = coef_cp * U_data_cp + (1.0 - coef_cp) * (
                    self.U_data + dt * dU)
                self._update_uic(dt, dUIC, stage=stage)
                self.limit()
                self._decompose(self.idx)
                if self.M > 2:
                    self._backup_uic_cp(self.idx_new)
                self.x_I = self.x_I_new
                self.idx = self.idx_new
                self._mark(self.point - self.x_I)
                self._phase_info(self.x_I)
                self._interface_info(idx=self.idx)

            # ---- stage 3 -------------------------------------------------
            if self.M > 2:
                self._compose()
                u_star = self._interface_velocity()
                dU, dUIC, _ = self.rhs()
                self.x_I_new = (1.0 / 3.0) * x_I_cp + (2.0 / 3.0) * (
                    self.x_I + u_star * dt)
                self.phi = self.point - self.x_I_new
                self._mark(self.phi, new=True)
                self._phase_info(self.x_I_new, new=True)
                self._interface_info(new=True, idx=self.idx)
                self.U_data = (1.0 / 3.0) * U_data_cp + (2.0 / 3.0) * (
                    self.U_data + dt * dU)
                self._update_uic(dt, dUIC, stage="3_3")
                self.limit()
                self._decompose(self.idx)
                self.x_I = self.x_I_new
                self.idx = self.idx_new
                self._mark(self.point - self.x_I)

            # Update the current interface-element geometry and coefficients.
            self._phase_info(self.x_I)
            self._interface_info(idx=self.idx)
            self._compose()
            t += dt
            steps += 1

        rho, u, p = self._fields()
        return self._result(rho, u, p, dt, steps)

    def _fields(self):
        rho0, u0, p0 = self.gas0.primitive(self.U_data[:, 0, :, 0])
        rho1, u1, p1 = self.gas1.primitive(self.U_data[:, 0, :, 1])
        v0 = self.phase_vol[0, :]
        v1 = self.phase_vol[1, :]
        vtot = np.maximum(v0 + v1, self.eps)
        rho = rho0 * v0 / vtot + rho1 * v1 / vtot
        mom = rho0 * u0 * v0 / vtot + rho1 * u1 * v1 / vtot
        u = mom / np.maximum(rho, self.eps)
        p = p0 * v0 / vtot + p1 * v1 / vtot
        return rho, u, p

    def _result(self, rho, u, p, dt, steps):
        try:
            rho_e, u_e, p_e = exact_fields(
                self.cfg, self.ele_centroid, self.cfg["time"]["t_end"]
            )
        except Exception as exc:
            rho_e = u_e = p_e = None
            exact_error = str(exc)
        else:
            exact_error = None
        return {
            "x": self.ele_centroid,
            "rho": rho,
            "u": u,
            "p": p,
            "rho_exact": rho_e,
            "u_exact": u_e,
            "p_exact": p_e,
            "exact_error": exact_error,
            "dt": dt,
            "steps": steps,
            "interface": self.x_I,
        }


def solve_two_medium(cfg: dict) -> dict:
    return TwoMediumSolver(cfg).run()
