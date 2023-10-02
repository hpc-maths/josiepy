# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from josie.mesh.cellset import MeshCellSet
from .schemes import FourEqScheme
from .state import Q, FourEqConsFields, FourEqConsState
from josie.twofluid.fields import Phases

from .problem import FourEqProblem


# This is rigorously valid only in the case of
# linearized gas EOS according to 'Chanteperdix et al., 2002'
class Exact(FourEqScheme):
    problem: FourEqProblem

    def P0(self, alpha: np.ndarray):
        p0 = self.problem.eos[Phases.PHASE1].p0
        rho10 = self.problem.eos[Phases.PHASE1].rho0
        rho20 = self.problem.eos[Phases.PHASE2].rho0
        c1 = self.problem.eos[Phases.PHASE1].c0
        c2 = self.problem.eos[Phases.PHASE2].c0

        out = np.full_like(alpha, p0)
        ind = np.where(alpha > 0)
        out[ind] -= alpha[ind] * rho10 * c1**2
        ind = np.where(1 - alpha > 0)
        out[ind] -= (1.0 - alpha[ind]) * rho20 * c2**2

        return out

    def deltaU(
        self,
        Q_L: np.ndarray,
        Q_R: np.ndarray,
        P: np.ndarray,
        normals: np.ndarray,
    ):
        fields = Q.fields

        alpha_L = Q_L[..., fields.alpha]
        alpha_R = Q_R[..., fields.alpha]
        P_L = Q_L[..., fields.P]
        P_R = Q_R[..., fields.P]
        rho_L = Q_L[..., fields.rho]
        rho_R = Q_R[..., fields.rho]
        c_L = Q_L[..., fields.c]
        c_R = Q_R[..., fields.c]
        P0L = self.P0(alpha_L)
        P0R = self.P0(alpha_R)

        # To be modified for 2D where U is U dot n
        U_L = np.einsum(
            "...kl,...l->...k",
            np.stack((Q_L[..., fields.U], Q_L[..., fields.V]), axis=-1),
            normals,
        )
        U_R = np.einsum(
            "...kl,...l->...k",
            np.stack((Q_R[..., fields.U], Q_R[..., fields.V]), axis=-1),
            normals,
        )

        dU = U_L - U_R

        ind = np.where(P <= P_L)
        dU[ind] += c_L[ind] * np.log((P_L[ind] - P0L[ind]) / (P[ind] - P0L[ind]))
        ind = np.where(P > P_L)
        dU[ind] += -(P[ind] - P_L[ind]) / np.sqrt(rho_L[ind] * (P[ind] - P0L[ind]))

        ind = np.where(P <= P_R)
        dU[ind] -= -c_R[ind] * np.log((P_R[ind] - P0R[ind]) / (P[ind] - P0R[ind]))
        ind = np.where(P > P_R)
        dU[ind] -= (P[ind] - P_R[ind]) / np.sqrt(rho_R[ind] * (P[ind] - P0R[ind]))

        return dU

    def ddeltaU_dP(self, Q_L: np.ndarray, Q_R: np.ndarray, P: np.ndarray):
        fields = Q.fields

        # To be modified for 2D where U is U dot n
        alpha_L = Q_L[..., fields.alpha]
        alpha_R = Q_R[..., fields.alpha]
        P_L = Q_L[..., fields.P]
        P_R = Q_R[..., fields.P]
        rho_L = Q_L[..., fields.rho]
        rho_R = Q_R[..., fields.rho]
        c_L = Q_L[..., fields.c]
        c_R = Q_R[..., fields.c]
        P0L = self.P0(alpha_L)
        P0R = self.P0(alpha_R)

        ddU_dP = np.zeros_like(alpha_L)

        ind = np.where(P <= P_L)
        ddU_dP[ind] += c_L[ind] / (P0L[ind] - P[ind])
        ind = np.where(P > P_L)
        ddU_dP[ind] += (2.0 * P0L[ind] - P[ind] - P_L[ind]) / (
            2.0 * (P[ind] - P0L[ind]) * np.sqrt((P[ind] - P0L[ind]) * rho_L[ind])
        )

        ind = np.where(P <= P_R)
        ddU_dP[ind] += c_R[ind] / (P0R[ind] - P[ind])
        ind = np.where(P > P_R)
        ddU_dP[ind] += (2.0 * P0R[ind] - P[ind] - P_R[ind]) / (
            2.0 * (P[ind] - P0R[ind]) * np.sqrt((P[ind] - P0R[ind]) * rho_R[ind])
        )

        return ddU_dP

    def solvePressure(self, P_init: np.ndarray, Q_L: Q, Q_R: Q, normals: np.ndarray):
        P = P_init.copy()
        dP = np.zeros_like(P)
        tol = 1e-8
        firstLoop = True

        # No Newton algorithm where exact pressure equilibrium
        ind = np.where(self.deltaU(Q_L, Q_R, P, normals) != 0.0)[0]
        P0tilde = np.maximum(
            self.P0(Q_L[..., Q.fields.alpha]), self.P0(Q_R[..., Q.fields.alpha])
        )
        # Newton-Raphson loop
        while len(ind) > 0 or firstLoop:
            if firstLoop:
                firstLoop = False
            dP.fill(0)
            dP[ind, ...] = -self.deltaU(
                Q_L[ind, ...],
                Q_R[ind, ...],
                P[ind, ...],
                normals[ind, ...],
            ) / self.ddeltaU_dP(Q_L[ind, ...], Q_R[ind, ...], P[ind, ...])
            P[ind, ...] += np.maximum(
                dP[ind, ...], 0.9 * (P0tilde[ind, ...] - P[ind, ...])
            )

            ind = np.where(np.abs(dP / P) > tol)[0]

        return P

    def solve_RP(self, Q_L: Q, Q_R: Q, Qc: FourEqConsState, normals: np.ndarray):
        fields = Q.fields
        cfields = FourEqConsFields
        Qc_R = Q_R.view(Q).get_conservative()

        arho1_L = Q_L[..., fields.arho1]
        arho2_L = Q_L[..., fields.arho2]
        P_L = Q_L[..., fields.P]
        U_L = np.einsum(
            "...kl,...l->...k",
            np.stack((Q_L[..., fields.U], Q_L[..., fields.V]), axis=-1),
            normals,
        )
        c_L = Q_L[..., fields.c]
        alpha_L = Q_L[..., fields.alpha]
        rho_L = Q_L[..., fields.rho]
        P0_L = self.P0(alpha_L)

        arho1_R = Q_R[..., fields.arho1]
        arho2_R = Q_R[..., fields.arho2]
        P_R = Q_R[..., fields.P]
        U_R = np.einsum(
            "...kl,...l->...k",
            np.stack((Q_R[..., fields.U], Q_R[..., fields.V]), axis=-1),
            normals,
        )
        c_R = Q_R[..., fields.c]
        alpha_R = Q_R[..., fields.alpha]
        rho_R = Q_R[..., fields.rho]
        P0_R = self.P0(alpha_R)

        # Solve for Pstar
        # Could change the init pressure
        P0tilde = np.maximum(P0_L, P0_R)
        P_star = self.solvePressure(
            np.maximum(0.5 * (P_L + P_R), 1.1 * P0tilde), Q_L, Q_R, normals
        )

        # Compute Ustar
        U_star = U_L.copy()
        ind = np.where(P_star <= P_L)
        U_star[ind] += c_L[ind] * np.log(
            (P_L[ind] - P0_L[ind]) / (P_star[ind] - P0_L[ind])
        )

        ind = np.where(P_star > P_L)
        U_star[ind] -= (P_star[ind] - P_L[ind]) / np.sqrt(
            rho_L[ind] * (P_star[ind] - P0_L[ind])
        )

        # If 0 < Ustar
        #   If left shock
        #       If right of left shock -> Qc_L_star

        # If left shock
        arho1_L_star = arho1_L * (1 + (P_star - P_L) / (rho_L * c_L**2))
        arho2_L_star = arho2_L * (1 + (P_star - P_L) / (rho_L * c_L**2))
        rho_L_star = arho1_L_star + arho2_L_star
        rhoc_sq_L_star = (
            arho1_L_star * self.problem.eos[Phases.PHASE1].c0 ** 2
            + arho2_L_star * self.problem.eos[Phases.PHASE2].c0 ** 2
        )
        S_L = np.empty_like(U_L) * np.nan
        ind = np.where(P_star > P_L)
        S_L[ind] = (
            U_star[ind] - np.sqrt(rho_L[ind] / rhoc_sq_L_star[ind]) * c_L[ind] ** 2
        )
        # If left of left shock -> already done
        # If right of left shock -> Qc_L_star

        ind = np.where((0 < U_star) * (P_star > P_L) * (S_L < 0))
        ind_tmp = ind[:3]

        # TODO: to be changed for 2D
        Qc[ind_tmp + (cfields.arho,)] = alpha_L[ind] * rho_L_star[ind]
        Qc[ind_tmp + (cfields.rhoU,)] = np.einsum(
            "...k,...l->...kl", rho_L_star * U_star, normals
        )[ind_tmp + (0,)]
        Qc[ind + (cfields.rhoV,)] = np.einsum(
            "...k,...l->...kl", rho_L_star * U_star, normals
        )[ind_tmp + (1,)]
        Qc[ind_tmp + (cfields.arho1,)] = arho1_L_star[ind]
        Qc[ind_tmp + (cfields.arho2,)] = arho2_L_star[ind]

        #   If left fan -> check if in or out of the fan
        #       If left of the fan -> already done

        #       If in the fan -> Qc_L_fan
        SH_L = U_L - c_L
        ST_L = U_star - c_L
        arho1_L_fan = arho1_L * np.exp((U_L - c_L) / c_L)
        arho2_L_fan = arho2_L * np.exp((U_L - c_L) / c_L)
        rho_L_fan = arho1_L_fan + arho2_L_fan

        ind = np.where((0 < U_star) * (P_star <= P_L) * (SH_L < 0) * (ST_L > 0))
        ind_tmp = ind[:3]

        # TODO: to be changed for 2D
        Qc[ind_tmp + (cfields.arho,)] = alpha_L[ind] * rho_L_fan[ind]
        Qc[ind_tmp + (cfields.rhoU,)] = np.einsum(
            "...k,...l->...kl", rho_L_fan * c_L, normals
        )[ind_tmp + (0,)]
        Qc[ind_tmp + (cfields.rhoV,)] = np.einsum(
            "...k,...l->...kl", rho_L_fan * c_L, normals
        )[ind_tmp + (1,)]
        Qc[ind_tmp + (cfields.arho1,)] = arho1_L_fan[ind]
        Qc[ind_tmp + (cfields.arho2,)] = arho2_L_fan[ind]

        #       If right of the fan -> compute state
        arho1_L_star = arho1_L * np.exp((U_L - U_star) / c_L)
        arho2_L_star = arho2_L * np.exp((U_L - U_star) / c_L)
        rho_L_star = arho1_L_star + arho2_L_star

        ind = np.where((0 < U_star) * (P_star <= P_L) * (ST_L <= 0))
        ind_tmp = ind[:3]

        Qc[ind_tmp + (cfields.arho,)] = alpha_L[ind] * rho_L_star[ind]
        Qc[ind_tmp + (cfields.rhoU,)] = np.einsum(
            "...k,...l->...kl", rho_L_star * U_star, normals
        )[ind_tmp + (0,)]
        Qc[ind_tmp + (cfields.rhoV,)] = np.einsum(
            "...k,...l->...kl", rho_L_star * U_star, normals
        )[ind_tmp + (1,)]
        Qc[ind_tmp + (cfields.arho1,)] = arho1_L_star[ind]
        Qc[ind_tmp + (cfields.arho2,)] = arho2_L_star[ind]

        # If 0 > Ustar
        #   If right shock
        arho1_R_star = arho1_R * (1 + (P_star - P_R) / (rho_R * c_R**2))
        arho2_R_star = arho2_R * (1 + (P_star - P_R) / (rho_R * c_R**2))
        rho_R_star = arho1_R_star + arho2_R_star
        rhoc_sq_R_star = (
            arho1_R_star * self.problem.eos[Phases.PHASE1].c0 ** 2
            + arho2_R_star * self.problem.eos[Phases.PHASE2].c0 ** 2
        )
        S_star_R = np.empty_like(U_R) * np.nan
        ind = np.where(P_star > P_R)
        S_star_R[ind] = (
            U_star[ind] + np.sqrt(rho_R[ind] / rhoc_sq_R_star[ind]) * c_R[ind] ** 2
        )

        #   If right of right shock -> Qc_R
        ind = np.where((0 >= U_star) * (P_star > P_R) * (S_star_R < 0))
        Qc[ind] = Qc_R[ind]

        #   If left of right shock -> Qc_R_star
        ind = np.where((0 >= U_star) * (P_star > P_R) * (S_star_R >= 0))
        ind_tmp = ind[:3]

        # TODO: to be changed for 2D
        Qc[ind_tmp + (cfields.arho,)] = alpha_R[ind] * rho_R_star[ind]
        Qc[ind_tmp + (cfields.rhoU,)] = np.einsum(
            "...k,...l->...kl", rho_R_star * U_star, normals
        )[ind_tmp + (0,)]
        Qc[ind_tmp + (cfields.rhoV,)] = np.einsum(
            "...k,...l->...kl", rho_R_star * U_star, normals
        )[ind_tmp + (1,)]
        Qc[ind_tmp + (cfields.arho1,)] = arho1_R_star[ind]
        Qc[ind_tmp + (cfields.arho2,)] = arho2_R_star[ind]

        #   If right fan -> check if in or out of the fan
        SH_R = U_R + c_R
        ST_R = U_star + c_R
        #       If right of the fan -> Qc_R
        ind = np.where((0 >= U_star) * (P_star <= P_R) * (SH_R < 0))
        ind_tmp = ind[:3]

        Qc[ind] = Qc_R[ind]

        #       If in the fan -> Qc_R_fan
        arho1_R_fan = arho1_R * np.exp(-(U_R + c_R) / c_R)
        arho2_R_fan = arho2_R * np.exp(-(U_R + c_R) / c_R)
        rho_R_fan = arho1_R_fan + arho2_R_fan

        ind = np.where((0 >= U_star) * (P_star <= P_R) * (SH_R >= 0) * (ST_R < 0))
        ind_tmp = ind[:3]

        # TODO: to be changed for 2D
        Qc[ind_tmp + (cfields.arho,)] = alpha_R[ind] * rho_R_fan[ind]
        Qc[ind_tmp + (cfields.rhoU,)] = np.einsum(
            "...k,...l->...kl", -rho_R_fan * c_R, normals
        )[ind_tmp + (0,)]
        Qc[ind_tmp + (cfields.rhoV,)] = np.einsum(
            "...k,...l->...kl", -rho_R_fan * c_R, normals
        )[ind_tmp + (1,)]
        Qc[ind_tmp + (cfields.arho1,)] = arho1_R_fan[ind]
        Qc[ind_tmp + (cfields.arho2,)] = arho2_R_fan[ind]

        #       If left of the fan -> Qc_R_star
        arho1_R_star = arho1_R * np.exp(-(U_R - U_star) / c_R)
        arho2_R_star = arho2_R * np.exp(-(U_R - U_star) / c_R)
        rho_R_star = arho1_R_star + arho2_R_star

        ind = np.where((0 >= U_star) * (P_star <= P_R) * (ST_R >= 0))
        ind_tmp = ind[:3]

        Qc[ind_tmp + (cfields.arho,)] = alpha_R[ind] * rho_R_star[ind]
        Qc[ind_tmp + (cfields.rhoU,)] = np.einsum(
            "...k,...l->...kl", rho_R_star * U_star, normals
        )[ind_tmp + (0,)]
        Qc[ind_tmp + (cfields.rhoV,)] = np.einsum(
            "...k,...l->...kl", rho_R_star * U_star, normals
        )[ind_tmp + (1,)]
        Qc[ind_tmp + (cfields.arho1,)] = arho1_R_star[ind]
        Qc[ind_tmp + (cfields.arho2,)] = arho2_R_star[ind]

        return Qc

    def intercellFlux(
        self,
        Q_L: Q,
        Q_R: Q,
        normals: np.ndarray,
        surfaces: np.ndarray,
    ):
        r"""Exact solver scheme

        Parameters
        ----------
        cells:
            A :class:`MeshCellSet` containing the state of the mesh cells

        neighs
            A :class:`NeighboursCellSet` containing data of neighbour cells
            corresponding to the :attr:`values`

        Returns
        -------
        F
            The value of the numerical convective flux multiplied by the
            surface value :math:`\numConvective`
        """
        FS = np.zeros_like(Q_L).view(Q)

        # Prepare state
        Qc_L = Q_L.view(Q).get_conservative()
        Qc = Qc_L.copy()

        # Test if discontinuity
        ind = np.where(np.any(Q_L != Q_R, axis=-1))
        Qc[ind] = self.solve_RP(Q_L, Q_R, Qc.copy(), normals)[ind]

        # Compute flux
        intercells = Q_L.copy()
        intercells.view(Q).set_conservative(Qc)
        # TODO: Relaxation not necessary but it corrects arho
        # when it is slightly negative
        self.relaxation(intercells)
        self.auxilliaryVariableUpdate(intercells)
        F = np.einsum("...mkl,...l->...mk", self.problem.F(intercells), normals)

        # Multiply by surfaces
        FS.set_conservative(surfaces[..., np.newaxis, np.newaxis] * F)

        return FS

    def CFL(
        self,
        cells: MeshCellSet,
        CFL_value,
    ) -> float:
        dt = super().CFL(cells, CFL_value)

        dx = cells.min_length

        # Get the velocity components
        UV_slice = slice(Q.fields.U, Q.fields.V + 1)
        UV = cells.values[..., UV_slice]

        U = np.linalg.norm(UV, axis=-1)
        c = cells.values[..., Q.fields.c]

        sigma = np.max(np.abs(U) + c)

        dt = np.min((dt, CFL_value * dx / sigma))

        return dt
