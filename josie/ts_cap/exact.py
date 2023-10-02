# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from .schemes import TsCapScheme
from .state import Q, TsCapConsFields, TsCapConsState
from josie.twofluid.fields import Phases


# This is rigorously valid only in the case of
# linearized gas EOS according to 'Chanteperdix et al., 2002'
class Exact(TsCapScheme):
    def P0(self, abar: np.ndarray):
        p0 = self.problem.eos[Phases.PHASE1].p0
        rho10 = self.problem.eos[Phases.PHASE1].rho0
        rho20 = self.problem.eos[Phases.PHASE2].rho0
        c1 = self.problem.eos[Phases.PHASE1].c0
        c2 = self.problem.eos[Phases.PHASE2].c0

        out = np.full_like(abar, p0)
        ind = np.where(abar > 0)
        out[ind] -= abar[ind] * rho10 * c1**2
        ind = np.where(1 - abar > 0)
        out[ind] -= (1.0 - abar[ind]) * rho20 * c2**2

        return out

    def deltaU(
        self,
        Q_L: np.ndarray,
        Q_R: np.ndarray,
        P: np.ndarray,
        normals: np.ndarray,
    ):
        fields = Q.fields

        abar_L = Q_L[..., fields.abar]
        abar_R = Q_R[..., fields.abar]
        ad_L = Q_L[..., fields.ad]
        ad_R = Q_R[..., fields.ad]
        P_L = Q_L[..., fields.pbar]
        P_R = Q_R[..., fields.pbar]
        rho_L = Q_L[..., fields.rho]
        rho_R = Q_R[..., fields.rho]
        c_L = Q_L[..., fields.cFd]
        c_R = Q_R[..., fields.cFd]
        P0L = self.P0(abar_L)
        P0R = self.P0(abar_R)

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
        dU[ind] += (
            c_L[ind]
            * (1 - ad_L[ind])
            * np.log((P_L[ind] - P0L[ind]) / (P[ind] - P0L[ind]))
        )
        ind = np.where(P > P_L)
        dU[ind] += (
            -np.sqrt(1 - ad_L[ind])
            * (P[ind] - P_L[ind])
            / np.sqrt(rho_L[ind] * (P[ind] - P0L[ind]))
        )

        ind = np.where(P <= P_R)
        dU[ind] -= (
            -c_R[ind]
            * (1 - ad_R[ind])
            * np.log((P_R[ind] - P0R[ind]) / (P[ind] - P0R[ind]))
        )
        ind = np.where(P > P_R)
        dU[ind] -= (
            np.sqrt(1 - ad_R[ind])
            * (P[ind] - P_R[ind])
            / np.sqrt(rho_R[ind] * (P[ind] - P0R[ind]))
        )

        return dU

    def ddeltaU_dP(self, Q_L: np.ndarray, Q_R: np.ndarray, P: np.ndarray):
        fields = Q.fields

        # To be modified for 2D where U is U dot n
        abar_L = Q_L[..., fields.abar]
        abar_R = Q_R[..., fields.abar]
        ad_L = Q_L[..., fields.ad]
        ad_R = Q_R[..., fields.ad]
        P_L = Q_L[..., fields.pbar]
        P_R = Q_R[..., fields.pbar]
        rho_L = Q_L[..., fields.rho]
        rho_R = Q_R[..., fields.rho]
        c_L = Q_L[..., fields.cFd]
        c_R = Q_R[..., fields.cFd]
        P0L = self.P0(abar_L)
        P0R = self.P0(abar_R)

        ddU_dP = np.zeros_like(abar_L)

        ind = np.where(P <= P_L)
        ddU_dP[ind] += c_L[ind] * (1 - ad_L[ind]) / (P0L[ind] - P[ind])
        ind = np.where(P > P_L)
        ddU_dP[ind] += (
            np.sqrt(1 - ad_L[ind])
            * (2.0 * P0L[ind] - P[ind] - P_L[ind])
            / (2.0 * (P[ind] - P0L[ind]) * np.sqrt((P[ind] - P0L[ind]) * rho_L[ind]))
        )

        ind = np.where(P <= P_R)
        ddU_dP[ind] += c_R[ind] * (1 - ad_R[ind]) / (P0R[ind] - P[ind])
        ind = np.where(P > P_R)
        ddU_dP[ind] += (
            np.sqrt(1 - ad_R[ind])
            * (2.0 * P0R[ind] - P[ind] - P_R[ind])
            / (2.0 * (P[ind] - P0R[ind]) * np.sqrt((P[ind] - P0R[ind]) * rho_R[ind]))
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
            self.P0(Q_L[..., Q.fields.abar]), self.P0(Q_R[..., Q.fields.abar])
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

        arho1_L = Q_L[..., fields.arho1]
        arho2_L = Q_L[..., fields.arho2]
        arho1d_L = Q_L[..., fields.arho1d]
        P_L = Q_L[..., fields.pbar]
        U_L = np.einsum(
            "...kl,...l->...k",
            np.stack((Q_L[..., fields.U], Q_L[..., fields.V]), axis=-1),
            normals,
        )
        c_L = Q_L[..., fields.cFd]
        abar_L = Q_L[..., fields.abar]
        ad_L = Q_L[..., fields.ad]
        rho_L = Q_L[..., fields.rho]
        P0_L = self.P0(abar_L)

        arho1_R = Q_R[..., fields.arho1]
        arho2_R = Q_R[..., fields.arho2]
        arho1d_R = Q_R[..., fields.arho1d]
        P_R = Q_R[..., fields.pbar]
        U_R = np.einsum(
            "...kl,...l->...k",
            np.stack((Q_R[..., fields.U], Q_R[..., fields.V]), axis=-1),
            normals,
        )
        c_R = Q_R[..., fields.cFd]
        abar_R = Q_R[..., fields.abar]
        ad_R = Q_R[..., fields.ad]
        rho_R = Q_R[..., fields.rho]
        P0_R = self.P0(abar_R)

        # Solve for Pstar
        # Could change the init pressure
        P0tilde = np.maximum(P0_L, P0_R)
        P_star = self.solvePressure(
            np.maximum(0.5 * (P_L + P_R), 1.1 * P0tilde), Q_L, Q_R, normals
        )

        # Compute Ustar
        U_star = U_L.copy()
        ind = np.where(P_star <= P_L)
        U_star[ind] += (
            c_L[ind]
            * (1 - ad_L[ind])
            * np.log((P_L[ind] - P0_L[ind]) / (P_star[ind] - P0_L[ind]))
        )

        ind = np.where(P_star > P_L)
        U_star[ind] -= (
            np.sqrt(1 - ad_L[ind])
            * (P_star[ind] - P_L[ind])
            / np.sqrt(rho_L[ind] * (P_star[ind] - P0_L[ind]))
        )

        # If 0 < Ustar
        #   If left shock
        #       If right of left shock -> Qc_L_star

        # If left shock
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
        ind = np.where(P_star > P_L)
        S_L[ind] = U_star[ind] + (U_L[ind] - U_star[ind]) / (1 - r[ind])
        # If left of left shock -> already done
        # If right of left shock -> Qc_L_star

        ind = np.where((0 < U_star) * (P_star > P_L) * (S_L < 0))
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
        r = 1 + (1 - ad_R) / (ad_R + (rho_R * c_R**2 * (1 - ad_R)) / (P_star - P_R))
        arho1_R_star = arho1_R * r
        arho2_R_star = arho2_R * r
        arho1d_R_star = arho1d_R * r
        ad_R_star = ad_R * r
        rho_R_star = arho1_R_star + arho2_R_star + arho1d_R_star

        S_star_R = np.empty_like(U_R) * np.nan
        ind = np.where(P_star > P_R)
        S_star_R[ind] = U_star[ind] + (U_R[ind] - U_star[ind]) / (1 - r[ind])

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
        self.auxilliaryVariableUpdate(intercells)
        F = np.einsum("...mkl,...l->...mk", self.problem.F(intercells), normals)

        # Multiply by surfaces
        FS.set_conservative(surfaces[..., np.newaxis, np.newaxis] * F)

        return FS
