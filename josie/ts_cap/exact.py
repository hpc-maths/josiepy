# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from .schemes import TsCapScheme
from .state import Q, TsCapConsFields, TsCapConsState
from josie.twofluid.fields import Phases
from ..dimension import MAX_DIMENSIONALITY

from .eos import TwoPhaseEOS


# This is rigorously valid only in the case of
# linearized gas EOS according to 'Chanteperdix et al., 2002'
class ExactHyp(TsCapScheme):
    def post_init(self, cells):
        super().post_init(cells)

        self.dU = np.zeros_like(cells.values[..., 0])
        self.ddU_dP = np.zeros_like(cells.values[..., 0])

    def post_extrapolation(self, values: Q):
        # self.prim2Q(values)
        self.relaxation(values)

        # auxilliary variables update
        self.auxilliaryVariableUpdateNoGeo(values)

    def P0(self, abar: np.ndarray):
        p0 = self.problem.eos[Phases.PHASE1].p0
        rho10 = self.problem.eos[Phases.PHASE1].rho0
        rho20 = self.problem.eos[Phases.PHASE2].rho0
        c1 = self.problem.eos[Phases.PHASE1].c0
        c2 = self.problem.eos[Phases.PHASE2].c0

        self.P0_ = np.where(abar > 0, p0 - abar * rho10 * c1**2, p0)
        self.P0_ = np.where(
            1 - abar > 0, self.P0_ - (1.0 - abar) * rho20 * c2**2, self.P0_
        )

        return self.P0_

    def deltaU(
        self,
        Q_L: np.ndarray,
        Q_R: np.ndarray,
        P: np.ndarray,
        dUn: np.ndarray,
        P0L: np.ndarray,
        P0R: np.ndarray,
    ):
        fields = Q.fields

        ad_L = Q_L[..., fields.ad]
        ad_R = Q_R[..., fields.ad]
        P_L = Q_L[..., fields.pbar]
        P_R = Q_R[..., fields.pbar]
        rho_L = Q_L[..., fields.rho]
        rho_R = Q_R[..., fields.rho]
        c_L = Q_L[..., fields.cFd]
        c_R = Q_R[..., fields.cFd]

        self.dU = dUn.copy()

        self.dU += np.where(
            P <= P_L,
            c_L * (1 - ad_L) * np.log((P_L - P0L) / (P - P0L)),
            -np.sqrt(1 - ad_L) * (P - P_L) / np.sqrt(rho_L * (P - P0L)),
        )
        self.dU -= np.where(
            P <= P_R,
            -c_R * (1 - ad_R) * np.log((P_R - P0R) / (P - P0R)),
            np.sqrt(1 - ad_R) * (P - P_R) / np.sqrt(rho_R * (P - P0R)),
        )

        return self.dU

    def ddeltaU_dP(
        self,
        Q_L: np.ndarray,
        Q_R: np.ndarray,
        P: np.ndarray,
        P0L: np.ndarray,
        P0R: np.ndarray,
    ):
        fields = Q.fields

        # To be modified for 2D where U is U dot n
        ad_L = Q_L[..., fields.ad]
        ad_R = Q_R[..., fields.ad]
        P_L = Q_L[..., fields.pbar]
        P_R = Q_R[..., fields.pbar]
        rho_L = Q_L[..., fields.rho]
        rho_R = Q_R[..., fields.rho]
        c_L = Q_L[..., fields.cFd]
        c_R = Q_R[..., fields.cFd]

        self.ddU_dP = np.where(
            P <= P_L,
            c_L * (1 - ad_L) / (P0L - P),
            (
                np.sqrt(1 - ad_L)
                * (2.0 * P0L - P - P_L)
                / (2.0 * (P - P0L) * np.sqrt((P - P0L) * rho_L))
            ),
        )
        self.ddU_dP += np.where(
            P <= P_R,
            c_R * (1 - ad_R) / (P0R - P),
            (
                np.sqrt(1 - ad_R)
                * (2.0 * P0R - P - P_R)
                / (2.0 * (P - P0R) * np.sqrt((P - P0R) * rho_R))
            ),
        )

        return self.ddU_dP

    def solvePressure(
        self,
        P_init: np.ndarray,
        Q_L: Q,
        Q_R: Q,
        dUn: np.ndarray,
        P0L: np.ndarray,
        P0R: np.ndarray,
    ):
        P = P_init
        tol = 1e-8
        firstLoop = True

        # No Newton algorithm where exact pressure equilibrium
        ind = np.where(self.deltaU(Q_L, Q_R, P, dUn, P0L, P0R) != 0.0)[0]
        P0tilde = np.maximum(
            self.P0(Q_L[..., Q.fields.abar]), self.P0(Q_R[..., Q.fields.abar])
        )

        # Newton-Raphson loop
        while len(ind) > 0 or firstLoop:
            if firstLoop:
                firstLoop = False
            self.dP = -self.deltaU(
                Q_L,
                Q_R,
                P,
                dUn,
                P0L,
                P0R,
            ) / self.ddeltaU_dP(Q_L, Q_R, P, P0L, P0R)
            P += np.maximum(self.dP, 0.9 * (P0tilde - P))

            ind = np.where(np.abs(self.dP / P) > tol)[0]

        return P

    @classmethod
    def solveAlpha1dFan(cls, RHS: np.ndarray, ad: np.ndarray):
        ad_fan_old = np.copy(ad)
        ad_fan_new = np.copy(ad)
        tol = 1e-8
        firstLoop = True

        ind = np.where(ad > 0)

        # Newton-Raphson loop
        while (
            firstLoop
            or (
                np.abs(ad_fan_new[ind] - ad_fan_old[ind])
                / (0.5 * (ad_fan_new[ind] + ad_fan_old[ind]))
                > tol
            ).any()
        ):
            if firstLoop:
                firstLoop = False

            ad_fan_old[ind] = ad_fan_new[ind]
            ad_fan_new[ind] = ad_fan_old[ind] - (
                cls.F_adFan(
                    ad_fan_old[ind],
                    ad[ind],
                )
                - RHS[ind]
            ) / cls.dF_adFan(
                ad_fan_old[ind],
            )

        return ad_fan_new

    @classmethod
    def F_adFan(cls, ad_fan: np.ndarray, ad: np.ndarray):
        return 1 / (1 - ad_fan) + np.log((ad_fan / ad) * (1 - ad) / (1 - ad_fan))

    @classmethod
    def dF_adFan(cls, ad_fan: np.ndarray):
        return 1 / (1 - ad_fan) ** 2 / ad_fan

    def solve_RP(self, Q_L: Q, Q_R: Q, Qc: TsCapConsState, normals: np.ndarray):
        fields = Q.fields
        cfields = TsCapConsFields
        Qc_R = Q_R.view(Q).get_conservative()

        # Left state
        arho1_L = Q_L[..., fields.arho1]
        arho2_L = Q_L[..., fields.arho2]
        arho1d_L = Q_L[..., fields.arho1d]
        P_L = Q_L[..., fields.pbar]
        U_L = np.einsum(
            "...l,...->...l", Q_L[..., fields.U], normals[..., 0]
        ) + np.einsum("...l,...->...l", Q_L[..., fields.V], normals[..., 1])
        c_L = Q_L[..., fields.cFd]
        abar_L = Q_L[..., fields.abar]
        ad_L = Q_L[..., fields.ad]
        rho_L = Q_L[..., fields.rho]
        P0L = self.P0(abar_L)

        # Right state
        arho1_R = Q_R[..., fields.arho1]
        arho2_R = Q_R[..., fields.arho2]
        arho1d_R = Q_R[..., fields.arho1d]
        P_R = Q_R[..., fields.pbar]
        U_R = np.einsum(
            "...l,...->...l", Q_R[..., fields.U], normals[..., 0]
        ) + np.einsum("...l,...->...l", Q_R[..., fields.V], normals[..., 1])
        c_R = Q_R[..., fields.cFd]
        abar_R = Q_R[..., fields.abar]
        ad_R = Q_R[..., fields.ad]
        rho_R = Q_R[..., fields.rho]
        P0R = self.P0(abar_R)

        # Solve for Pstar
        # Could change the init pressure
        P0tilde = np.maximum(P0L, P0R)
        P_star = self.solvePressure(
            np.maximum(0.5 * (P_L + P_R), P0tilde + 0.1 * np.abs(P0tilde)),
            Q_L,
            Q_R,
            U_L - U_R,
            P0L,
            P0R,
        )

        # Compute Ustar
        U_star = np.where(
            P_star <= P_L,
            U_L + c_L * (1 - ad_L) * np.log((P_L - P0L) / (P_star - P0L)),
            U_L - np.sqrt(1 - ad_L) * (P_star - P_L) / np.sqrt(rho_L * (P_star - P0L)),
        )

        # If 0 < Ustar
        #   If left shock
        #       If right of left shock -> Qc_L_star

        # If left shock
        ind = np.where(P_star > P_L)
        r = np.ones_like(ad_L) * np.nan
        r[ind] = 1 + (1 - ad_L[ind]) / (
            ad_L[ind]
            + (rho_L[ind] * c_L[ind] ** 2 * (1 - ad_L[ind])) / (P_star[ind] - P_L[ind])
        )
        arho1_L_star = arho1_L * r
        arho2_L_star = arho2_L * r
        ad_L_star = ad_L * r
        arho1d_L_star = arho1d_L * r
        rho_L_star = arho1_L_star + arho2_L_star + arho1d_L_star

        S_L = np.empty_like(U_L) * np.nan
        ind = np.where((P_star > P_L) & (r > 1))
        S_L[ind] = U_star[ind] + (U_L[ind] - U_star[ind]) / (1 - r[ind])
        ind = np.where((P_star > P_L) & (r == 1))
        S_L[ind] = U_star[ind] + (U_L[ind] - U_star[ind]) * (-np.inf)

        # If left of left shock -> already done
        # If right of left shock -> Qc_L_star

        ind = np.where((0 < U_star) & (P_star > P_L) & (S_L < 0))
        ind_tmp = ind[:3]

        # TODO: to be changed for 2D
        Qc[ind_tmp + (cfields.abarrho,)] = abar_L[ind] * rho_L_star[ind]
        Qc[ind_tmp + (cfields.rhoU,)] = np.einsum(
            "...k,...l->...kl", rho_L_star * U_star, normals
        )[ind_tmp + (0,)]
        Qc[ind + (cfields.rhoV,)] = np.einsum(
            "...k,...l->...kl", rho_L_star * U_star, normals
        )[ind_tmp + (1,)]
        Qc[ind_tmp + (cfields.ad,)] = ad_L_star[ind]
        Qc[ind_tmp + (cfields.arho1,)] = arho1_L_star[ind]
        Qc[ind_tmp + (cfields.arho2,)] = arho2_L_star[ind]
        Qc[ind_tmp + (cfields.arho1d,)] = arho1d_L_star[ind]

        #   If left fan -> check if in or out of the fan
        #       If left of the fan -> already done

        #       If in the fan -> Qc_L_fan
        ad_L_star = 1 - 1 / (
            1 + ad_L / (1 - ad_L) * np.exp((U_L - U_star) / c_L / (1 - ad_L))
        )
        SH_L = U_L - c_L
        ST_L = U_star - c_L * (1 - ad_L) / (1 - ad_L_star)

        ad_L_fan = np.ones_like(abar_L) * np.nan
        ad_L_fan[ind] = self.solveAlpha1dFan(
            U_L[ind] / c_L[ind] / (1 - ad_L[ind]), ad_L[ind]
        )
        arho1_L_fan = (
            (1 - ad_L_fan)
            * arho1_L
            / (1 - ad_L)
            * np.exp((U_L - (c_L * (1 - ad_L) / (1 - ad_L_fan))) / c_L / (1 - ad_L))
        )
        arho2_L_fan = (
            (1 - ad_L_fan)
            * arho2_L
            / (1 - ad_L)
            * np.exp((U_L - (c_L * (1 - ad_L) / (1 - ad_L_fan))) / c_L / (1 - ad_L))
        )
        arho1d_L_fan = (
            (1 - ad_L_fan)
            * arho1d_L
            / (1 - ad_L)
            * np.exp((U_L - (c_L * (1 - ad_L) / (1 - ad_L_fan))) / c_L / (1 - ad_L))
        )
        rho_L_fan = arho1_L_fan + arho2_L_fan + arho1d_L_fan

        ind = np.where((0 < U_star) * (P_star <= P_L) * (SH_L < 0) * (ST_L > 0))
        ind_tmp = ind[:3]

        # TODO: to be changed for 2D
        Qc[ind_tmp + (cfields.abarrho,)] = abar_L[ind] * rho_L_fan[ind]
        Qc[ind_tmp + (cfields.rhoU,)] = np.einsum(
            "...k,...l->...kl", rho_L_fan * c_L, normals
        )[ind_tmp + (0,)]
        Qc[ind_tmp + (cfields.rhoV,)] = np.einsum(
            "...k,...l->...kl", rho_L_fan * c_L, normals
        )[ind_tmp + (1,)]
        Qc[ind_tmp + (cfields.ad,)] = ad_L_fan[ind]
        Qc[ind_tmp + (cfields.arho1,)] = arho1_L_fan[ind]
        Qc[ind_tmp + (cfields.arho2,)] = arho2_L_fan[ind]
        Qc[ind_tmp + (cfields.arho1d,)] = arho1d_L_fan[ind]

        #       If right of the fan -> compute state
        arho1_L_star = (
            (1 - ad_L_star)
            * arho1_L
            / (1 - ad_L)
            * np.exp((U_L - U_star) / c_L / (1 - ad_L))
        )
        arho2_L_star = (
            (1 - ad_L_star)
            * arho2_L
            / (1 - ad_L)
            * np.exp((U_L - U_star) / c_L / (1 - ad_L))
        )
        arho1d_L_star = (
            (1 - ad_L_star)
            * arho1d_L
            / (1 - ad_L)
            * np.exp((U_L - U_star) / c_L / (1 - ad_L))
        )
        rho_L_star = arho1_L_star + arho2_L_star + arho1d_L_star

        ind = np.where((0 < U_star) * (P_star <= P_L) * (ST_L <= 0))
        ind_tmp = ind[:3]

        Qc[ind_tmp + (cfields.abarrho,)] = abar_L[ind] * rho_L_star[ind]
        Qc[ind_tmp + (cfields.rhoU,)] = np.einsum(
            "...k,...l->...kl", rho_L_star * U_star, normals
        )[ind_tmp + (0,)]
        Qc[ind_tmp + (cfields.rhoV,)] = np.einsum(
            "...k,...l->...kl", rho_L_star * U_star, normals
        )[ind_tmp + (1,)]
        Qc[ind_tmp + (cfields.ad,)] = ad_L_star[ind]
        Qc[ind_tmp + (cfields.arho1,)] = arho1_L_star[ind]
        Qc[ind_tmp + (cfields.arho2,)] = arho2_L_star[ind]
        Qc[ind_tmp + (cfields.arho1d,)] = arho1d_L_star[ind]

        # If 0 > Ustar
        #   If right shock
        r = np.full_like(ad_L, np.nan)
        ind = np.where(P_star > P_R)
        r[ind] = 1 + (1 - ad_R[ind]) / (
            ad_R[ind]
            + (rho_R[ind] * c_R[ind] ** 2 * (1 - ad_R[ind])) / (P_star[ind] - P_R[ind])
        )
        arho1_R_star = arho1_R * r
        arho2_R_star = arho2_R * r
        arho1d_R_star = arho1d_R * r
        ad_R_star = ad_R * r
        rho_R_star = arho1_R_star + arho2_R_star + arho1d_R_star

        S_star_R = np.empty_like(U_R) * np.nan
        ind = np.where((P_star > P_R) & (r > 1))
        S_star_R[ind] = U_star[ind] + (U_R[ind] - U_star[ind]) / (1 - r[ind])
        ind = np.where((P_star > P_R) & (r == 1))
        S_star_R[ind] = U_star[ind] + (U_R[ind] - U_star[ind]) / (-np.inf)

        #   If right of right shock -> Qc_R
        ind = np.where((0 >= U_star) * (P_star > P_R) * (S_star_R < 0))
        Qc[ind] = Qc_R[ind]

        #   If left of right shock -> Qc_R_star
        ind = np.where((0 >= U_star) * (P_star > P_R) * (S_star_R >= 0))
        ind_tmp = ind[:3]

        # TODO: to be changed for 2D
        Qc[ind_tmp + (cfields.abarrho,)] = abar_R[ind] * rho_R_star[ind]
        Qc[ind_tmp + (cfields.rhoU,)] = np.einsum(
            "...k,...l->...kl", rho_R_star * U_star, normals
        )[ind_tmp + (0,)]
        Qc[ind_tmp + (cfields.rhoV,)] = np.einsum(
            "...k,...l->...kl", rho_R_star * U_star, normals
        )[ind_tmp + (1,)]
        Qc[ind_tmp + (cfields.ad,)] = ad_R_star[ind]
        Qc[ind_tmp + (cfields.arho1,)] = arho1_R_star[ind]
        Qc[ind_tmp + (cfields.arho2,)] = arho2_R_star[ind]
        Qc[ind_tmp + (cfields.arho1d,)] = arho1d_R_star[ind]

        #   If right fan -> check if in or out of the fan
        ad_R_star = 1 - 1 / (
            1 + ad_R / (1 - ad_R) * np.exp(-(U_R - U_star) / c_R / (1 - ad_R))
        )
        SH_R = U_R + c_R
        ST_R = U_star + c_R * (1 - ad_R) / (1 - ad_R_star)
        #       If right of the fan -> Qc_R
        ind = np.where((0 >= U_star) * (P_star <= P_R) * (SH_R < 0))
        ind_tmp = ind[:3]

        Qc[ind] = Qc_R[ind]

        #       If in the fan -> Qc_R_fan
        ad_R_fan = np.ones_like(abar_R) * np.nan
        ad_R_fan[ind] = self.solveAlpha1dFan(
            -U_R[ind] / c_R[ind] / (1 - ad_R[ind]), ad_R[ind]
        )
        arho1_R_fan = (
            (1 - ad_R_fan)
            * arho1_R
            / (1 - ad_R)
            * np.exp(-(U_R + (c_R * (1 - ad_R) / (1 - ad_R_fan))) / c_R / (1 - ad_R))
        )
        arho2_R_fan = (
            (1 - ad_R_fan)
            * arho2_R
            / (1 - ad_R)
            * np.exp(-(U_R + (c_R * (1 - ad_R) / (1 - ad_R_fan))) / c_R / (1 - ad_R))
        )
        arho1d_R_fan = (
            (1 - ad_R_fan)
            * arho1d_R
            / (1 - ad_R)
            * np.exp(-(U_R + (c_R * (1 - ad_R) / (1 - ad_R_fan))) / c_R / (1 - ad_R))
        )
        rho_R_fan = arho1_R_fan + arho2_R_fan + arho1d_R_fan

        ind = np.where((0 >= U_star) * (P_star <= P_R) * (SH_R >= 0) * (ST_R < 0))
        ind_tmp = ind[:3]

        # TODO: to be changed for 2D
        Qc[ind_tmp + (cfields.abarrho,)] = abar_R[ind] * rho_R_fan[ind]
        Qc[ind_tmp + (cfields.rhoU,)] = np.einsum(
            "...k,...l->...kl", -rho_R_fan * c_R, normals
        )[ind_tmp + (0,)]
        Qc[ind_tmp + (cfields.rhoV,)] = np.einsum(
            "...k,...l->...kl", -rho_R_fan * c_R, normals
        )[ind_tmp + (1,)]
        Qc[ind_tmp + (cfields.ad,)] = ad_R_fan[ind]
        Qc[ind_tmp + (cfields.arho1,)] = arho1_R_fan[ind]
        Qc[ind_tmp + (cfields.arho2,)] = arho2_R_fan[ind]
        Qc[ind_tmp + (cfields.arho1d,)] = arho1d_R_fan[ind]

        #       If left of the fan -> Qc_R_star
        arho1_R_star = (
            (1 - ad_R_star)
            * arho1_R
            / (1 - ad_R)
            * np.exp(-(U_R - U_star) / c_R / (1 - ad_R))
        )
        arho2_R_star = (
            (1 - ad_R_star)
            * arho2_R
            / (1 - ad_R)
            * np.exp(-(U_R - U_star) / c_R / (1 - ad_R))
        )
        arho1d_R_star = (
            (1 - ad_R_star)
            * arho1d_R
            / (1 - ad_R)
            * np.exp(-(U_R - U_star) / c_R / (1 - ad_R))
        )
        rho_R_star = arho1_R_star + arho2_R_star + arho1d_R_star

        ind = np.where((0 >= U_star) * (P_star <= P_R) * (ST_R >= 0))
        ind_tmp = ind[:3]

        Qc[ind_tmp + (cfields.abarrho,)] = abar_R[ind] * rho_R_star[ind]
        Qc[ind_tmp + (cfields.rhoU,)] = np.einsum(
            "...k,...l->...kl", rho_R_star * U_star, normals
        )[ind_tmp + (0,)]
        Qc[ind_tmp + (cfields.rhoV,)] = np.einsum(
            "...k,...l->...kl", rho_R_star * U_star, normals
        )[ind_tmp + (1,)]
        Qc[ind_tmp + (cfields.ad,)] = ad_R_star[ind]
        Qc[ind_tmp + (cfields.arho1,)] = arho1_R_star[ind]
        Qc[ind_tmp + (cfields.arho2,)] = arho2_R_star[ind]
        Qc[ind_tmp + (cfields.arho1d,)] = arho1d_R_star[ind]

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
        self.auxilliaryVariableUpdateNoGeo(intercells)
        F = np.einsum("...mkl,...l->...mk", self.problem.F_hyper(intercells), normals)

        # Multiply by surfaces
        FS.set_conservative(surfaces[..., np.newaxis, np.newaxis] * F)

        return FS
