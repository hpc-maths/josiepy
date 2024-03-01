# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from .schemes import TsCapScheme
from .state import Q
from josie.twofluid.fields import Phases


# This is rigorously valid only in the case of
# linearized gas EOS according to 'Chanteperdix et al., 2002'
class ExactHyp(TsCapScheme):
    def post_init(self, cells):
        super().post_init(cells)

    def post_extrapolation(self, values: Q):
        self.prim2Qc(values)
        self.noMassTransfer = True
        self.relaxation(values)
        self.noMassTransfer = False

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

        return (
            dUn
            + np.where(
                P <= P_L,
                c_L * (1 - ad_L) * np.log((P_L - P0L) / (P - P0L)),
                -np.sqrt(1 - ad_L) * (P - P_L) / np.sqrt(rho_L * (P - P0L)),
            )
            - np.where(
                P <= P_R,
                -c_R * (1 - ad_R) * np.log((P_R - P0R) / (P - P0R)),
                np.sqrt(1 - ad_R) * (P - P_R) / np.sqrt(rho_R * (P - P0R)),
            )
        )

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

        return np.where(
            P <= P_L,
            c_L * (1 - ad_L) / (P0L - P),
            (
                np.sqrt(1 - ad_L)
                * (2.0 * P0L - P - P_L)
                / (2.0 * (P - P0L) * np.sqrt((P - P0L) * rho_L))
            ),
        ) + np.where(
            P <= P_R,
            c_R * (1 - ad_R) / (P0R - P),
            (
                np.sqrt(1 - ad_R)
                * (2.0 * P0R - P - P_R)
                / (2.0 * (P - P0R) * np.sqrt((P - P0R) * rho_R))
            ),
        )

    def d2deltaU_dP2(
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

        return np.where(
            P <= P_L,
            c_L * (1 - ad_L) / (P0L - P) ** 2,
            (
                np.sqrt(1 - ad_L)
                * (-4.0 * P0L + P + 3.0 * P_L)
                / (4.0 * (P - P0L) ** 2 * np.sqrt((P - P0L) * rho_L))
            ),
        ) + np.where(
            P <= P_R,
            c_R * (1 - ad_R) / (P0R - P) ** 2,
            (
                np.sqrt(1 - ad_R)
                * (-4.0 * P0R + P + 3 * P_R)
                / (4.0 * (P - P0R) ** 2 * np.sqrt((P - P0R) * rho_R))
            ),
        )

    def solvePressure(
        self,
        P_init: np.ndarray,
        Q_L: Q,
        Q_R: Q,
        dUn: np.ndarray,
        Un_L: np.ndarray,
        P0L: np.ndarray,
        P0R: np.ndarray,
    ):
        P = P_init
        tol = 1e-14
        firstLoop = True

        if np.any(P_init <= P0L):
            exit()

        P0tilde = np.maximum(P0L, P0R)
        self.dP = np.zeros_like(P)

        # Newton-Raphson loop
        k = 0
        fields = Q.fields
        F = np.zeros_like(dUn)
        while (
            np.any((np.abs(self.dP / P) > tol) & (np.abs(F) > tol * np.abs(Un_L)))
            or firstLoop
        ):
            k += 1
            if np.any(P <= P0L) or np.any(Q_L[..., fields.pbar] <= P0L):
                print(k)
                exit()

            if firstLoop:
                firstLoop = False

            F = self.deltaU(
                Q_L,
                Q_R,
                P,
                dUn,
                P0L,
                P0R,
            )
            dF = self.ddeltaU_dP(Q_L, Q_R, P, P0L, P0R)
            ddF = self.d2deltaU_dP2(Q_L, Q_R, P, P0L, P0R)
            self.dP = -2 * F * dF / (2 * dF**2 - F * ddF)
            P += np.maximum(self.dP, 0.9 * (P0tilde - P))
            if k == 20:
                tol = 1e-6
            if k == 40:
                exit()
        return P

    @classmethod
    def solveAlpha1dFan(cls, RHS: np.ndarray, ad: np.ndarray):
        ad_fan = ad.copy()
        dad_fan = np.zeros_like(ad_fan)
        tol = 1e-10
        eps = 1e-16
        firstLoop = True

        ind = np.where((ad > eps) & (ad < 1 - eps))

        # Newton-Raphson loop
        while len(ind[0]) or firstLoop:
            if firstLoop:
                firstLoop = False

            dad_fan.fill(0)
            dad_fan[ind] = -(
                cls.F_adFan(
                    ad_fan[ind],
                    ad[ind],
                )
                - RHS[ind]
            ) / cls.dF_adFan(
                ad_fan[ind],
            )
            dad_fan[ind] = np.where(
                dad_fan[ind] < 0,
                np.maximum(dad_fan[ind], -0.9 * ad_fan[ind]),
                np.minimum(dad_fan[ind], 0.9 * (1 - ad_fan[ind])),
            )

            ad_fan[ind] += dad_fan[ind]

            ind = np.where(
                (np.abs(np.divide(dad_fan, ad_fan, where=ad_fan > 0)) > tol) & (ad > 0)
            )

        return ad_fan

    @classmethod
    def F_adFan(cls, ad_fan: np.ndarray, ad: np.ndarray):
        return 1 / (1 - ad_fan) + np.log((ad_fan / ad) * (1 - ad) / (1 - ad_fan))

    @classmethod
    def dF_adFan(cls, ad_fan: np.ndarray):
        return 1 / (1 - ad_fan) ** 2 / ad_fan

    def solve_RP(self, Q_L: Q, Q_R: Q, intercells: Q, normals: np.ndarray):
        fields = Q.fields
        # Left state
        arho1_L = Q_L[..., fields.arho1]
        arho2_L = Q_L[..., fields.arho2]
        arho1d_L = Q_L[..., fields.arho1d]
        capSigma_L = Q_L[..., fields.capSigma]
        P_L = Q_L[..., fields.pbar]
        U_L = Q_L[..., fields.U]
        V_L = Q_L[..., fields.V]
        Un_L = np.einsum("...l,...->...l", U_L, normals[..., 0]) + np.einsum(
            "...l,...->...l", V_L, normals[..., 1]
        )
        c_L = Q_L[..., fields.cFd]
        abar_L = Q_L[..., fields.abar]
        ad_L = Q_L[..., fields.ad]
        rho_L = Q_L[..., fields.rho]
        P0L = self.P0(abar_L)

        # Right state
        arho1_R = Q_R[..., fields.arho1]
        arho2_R = Q_R[..., fields.arho2]
        arho1d_R = Q_R[..., fields.arho1d]
        capSigma_R = Q_R[..., fields.capSigma]
        P_R = Q_R[..., fields.pbar]
        U_R = Q_R[..., fields.U]
        V_R = Q_R[..., fields.V]
        Un_R = np.einsum("...l,...->...l", U_R, normals[..., 0]) + np.einsum(
            "...l,...->...l", V_R, normals[..., 1]
        )
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
            Un_L - Un_R,
            Un_L,
            P0L,
            P0R,
        )

        # Compute Ustar
        U_star = np.where(
            P_star <= P_L,
            Un_L + c_L * (1 - ad_L) * np.log((P_L - P0L) / (P_star - P0L)),
            Un_L - np.sqrt(1 - ad_L) * (P_star - P_L) / np.sqrt(rho_L * (P_star - P0L)),
        )
        # if np.any((np.abs(U_star) > 10 * np.abs(0.5 * (Un_L+Un_R))) & (np.abs(Un_L)>1e-2)):
        #    index = np.where((np.abs(U_star) > 10 * np.abs(0.5 * (Un_L+Un_R))) & (np.abs(Un_L)>1e-2))
        #    P_init = np.ravel(np.maximum(0.5 * (P_L + P_R), P0tilde + 0.1 * np.abs(P0tilde))[index])[0]
        #    dU = np.ravel((Un_L - Un_R)[index])[0]
        #    P0L = np.ravel(P0L[index])[0]
        #    P0R = np.ravel(P0R[index])[0]
        #    U_star = np.ravel(U_star[index])[0]
        #    Un_L = np.ravel(Un_L[index])[0]
        #    Un_R = np.ravel(Un_R[index])[0]
        #    P_star = np.ravel(P_star[index])[0]
        #    P_L = np.ravel(Q_L[..., fields.pbar][index])[0]
        #    P_R = np.ravel(Q_R[..., fields.pbar][index])[0]
        #    print(P_init)
        #    print(dU)
        #    print(U_star)
        #    print(Un_L)
        #    print(Un_R)
        #    print(P_star)
        #    print(P_L)
        #    print(P_R)
        #    print(P0L)
        #    print(P0R)
        #    exit()

        # If 0 < Ustar
        #   If left shock
        ind = np.where(P_star > P_L)
        r = np.ones_like(ad_L) * np.nan
        r[ind] = 1 + (1 - ad_L[ind]) / (
            ad_L[ind]
            + (rho_L[ind] * c_L[ind] ** 2 * (1 - ad_L[ind])) / (P_star[ind] - P_L[ind])
        )
        arho1_L_star = arho1_L * r
        arho2_L_star = arho2_L * r
        ad_L_star = ad_L * r
        arho1d_L_star = np.where(
            arho1d_L > 0, np.divide(ad_L_star * arho1d_L, ad_L, where=ad_L > 0), 0
        )
        capSigma_L_star = np.where(
            capSigma_L > 0, np.divide(ad_L_star * capSigma_L, ad_L, where=ad_L > 0), 0
        )
        rho_L_star = arho1_L_star + arho2_L_star + arho1d_L_star

        S_L = np.ones_like(Un_L) * np.nan
        ind = np.where((P_star > P_L) & (r > 1))
        S_L[ind] = U_star[ind] + (Un_L[ind] - U_star[ind]) / (1 - r[ind])
        ind = np.where((P_star > P_L) & (r == 1))
        S_L[ind] = U_star[ind] + (Un_L[ind] - U_star[ind]) * (-np.inf)

        # If left of left shock -> already done
        # If right of left shock -> Qc_L_star

        ind = np.where((0 < U_star) & (P_star > P_L) & (S_L < 0))
        ind_tmp = ind[:3]

        # TODO: to be changed for 2D
        intercells[ind_tmp + (fields.abarrho,)] = abar_L[ind] * rho_L_star[ind]
        intercells[ind_tmp + (fields.rhoU,)] = rho_L_star[ind] * (
            U_L[ind]
            + np.einsum("...k,...l->...kl", U_star - Un_L, normals)[ind_tmp + (0,)]
        )
        intercells[ind + (fields.rhoV,)] = rho_L_star[ind] * (
            V_L[ind]
            + np.einsum("...k,...l->...kl", U_star - Un_L, normals)[ind_tmp + (1,)]
        )
        intercells[ind_tmp + (fields.ad,)] = ad_L_star[ind]
        intercells[ind_tmp + (fields.arho1,)] = arho1_L_star[ind]
        intercells[ind_tmp + (fields.arho2,)] = arho2_L_star[ind]
        intercells[ind_tmp + (fields.arho1d,)] = arho1d_L_star[ind]
        intercells[ind_tmp + (fields.capSigma,)] = capSigma_L_star[ind]

        #   If left fan -> check if in or out of the fan
        #       If left of the fan -> already done

        #       If in the fan -> Qc_L_fan
        ad_L_star = 1 - 1 / (
            1 + ad_L / (1 - ad_L) * np.exp((Un_L - U_star) / c_L / (1 - ad_L))
        )
        SH_L = Un_L - c_L
        ST_L = U_star - c_L * (1 + ad_L * np.exp((Un_L - U_star) / c_L / (1 - ad_L)))

        ind = np.where((0 < U_star) * (P_star <= P_L) * (SH_L < 0) * (ST_L > 0))
        ind_tmp = ind[:3]

        ad_L_fan = np.ones_like(abar_L) * np.nan
        ad_L_fan[ind] = self.solveAlpha1dFan(
            Un_L[ind] / c_L[ind] / (1 - ad_L[ind]), ad_L[ind]
        )
        arho1_L_fan = (
            (1 - ad_L_fan)
            * arho1_L
            / (1 - ad_L)
            * np.exp((Un_L - (c_L * (1 - ad_L) / (1 - ad_L_fan))) / c_L / (1 - ad_L))
        )
        arho2_L_fan = (
            (1 - ad_L_fan)
            * arho2_L
            / (1 - ad_L)
            * np.exp((Un_L - (c_L * (1 - ad_L) / (1 - ad_L_fan))) / c_L / (1 - ad_L))
        )
        arho1d_L_fan = np.where(
            arho1d_L > 0, np.divide(ad_L_fan * arho1d_L, ad_L, where=ad_L > 0), 0
        )
        capSigma_L_fan = np.where(
            capSigma_L > 0, np.divide(ad_L_fan * capSigma_L, ad_L, where=ad_L > 0), 0
        )
        rho_L_fan = arho1_L_fan + arho2_L_fan + arho1d_L_fan

        # TODO: to be changed for 2D
        intercells[ind_tmp + (fields.abarrho,)] = abar_L[ind] * rho_L_fan[ind]
        intercells[ind_tmp + (fields.rhoU,)] = rho_L_fan[ind] * (
            U_L[ind]
            + np.einsum(
                "...k,...l->...kl", c_L * (1 - ad_L) / (1 - ad_L_fan) - Un_L, normals
            )[ind_tmp + (0,)]
        )
        intercells[ind_tmp + (fields.rhoV,)] = rho_L_fan[ind] * (
            V_L[ind]
            + np.einsum(
                "...k,...l->...kl", c_L * (1 - ad_L) / (1 - ad_L_fan) - Un_L, normals
            )[ind_tmp + (1,)]
        )
        intercells[ind_tmp + (fields.ad,)] = ad_L_fan[ind]
        intercells[ind_tmp + (fields.arho1,)] = arho1_L_fan[ind]
        intercells[ind_tmp + (fields.arho2,)] = arho2_L_fan[ind]
        intercells[ind_tmp + (fields.arho1d,)] = arho1d_L_fan[ind]
        intercells[ind_tmp + (fields.capSigma,)] = capSigma_L_fan[ind]

        #       If right of the fan -> compute state
        arho1_L_star = (
            (1 - ad_L_star)
            * arho1_L
            / (1 - ad_L)
            * np.exp((Un_L - U_star) / c_L / (1 - ad_L))
        )
        arho2_L_star = (
            (1 - ad_L_star)
            * arho2_L
            / (1 - ad_L)
            * np.exp((Un_L - U_star) / c_L / (1 - ad_L))
        )
        arho1d_L_star = np.where(
            arho1d_L > 0, np.divide(ad_L_star * arho1d_L, ad_L, where=ad_L > 0), 0
        )
        capSigma_L_star = np.where(
            capSigma_L > 0, np.divide(ad_L_star * capSigma_L, ad_L, where=ad_L > 0), 0
        )
        rho_L_star = arho1_L_star + arho2_L_star + arho1d_L_star

        ind = np.where((0 < U_star) * (P_star <= P_L) * (ST_L <= 0))
        ind_tmp = ind[:3]

        intercells[ind_tmp + (fields.abarrho,)] = abar_L[ind] * rho_L_star[ind]
        intercells[ind_tmp + (fields.rhoU,)] = rho_L_star[ind] * (
            U_L[ind]
            + np.einsum("...k,...l->...kl", U_star - Un_L, normals)[ind_tmp + (0,)]
        )
        intercells[ind + (fields.rhoV,)] = rho_L_star[ind] * (
            V_L[ind]
            + np.einsum("...k,...l->...kl", U_star - Un_L, normals)[ind_tmp + (1,)]
        )
        intercells[ind_tmp + (fields.ad,)] = ad_L_star[ind]
        intercells[ind_tmp + (fields.arho1,)] = arho1_L_star[ind]
        intercells[ind_tmp + (fields.arho2,)] = arho2_L_star[ind]
        intercells[ind_tmp + (fields.arho1d,)] = arho1d_L_star[ind]
        intercells[ind_tmp + (fields.capSigma,)] = capSigma_L_star[ind]

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
        ad_R_star = ad_R * r
        arho1d_R_star = np.where(
            arho1d_R > 0, np.divide(ad_R_star * arho1d_R, ad_R, where=ad_R > 0), 0
        )
        capSigma_R_star = np.where(
            capSigma_R > 0, np.divide(ad_R_star * capSigma_R, ad_R, where=ad_R > 0), 0
        )
        rho_R_star = arho1_R_star + arho2_R_star + arho1d_R_star

        S_star_R = np.ones_like(Un_R) * np.nan
        ind = np.where((P_star > P_R) & (r > 1))
        S_star_R[ind] = U_star[ind] + (Un_R[ind] - U_star[ind]) / (1 - r[ind])
        ind = np.where((P_star > P_R) & (r == 1))
        S_star_R[ind] = U_star[ind] + (Un_R[ind] - U_star[ind]) / (-np.inf)

        #   If right of right shock -> Qc_R
        ind = np.where((0 >= U_star) * (P_star > P_R) * (S_star_R < 0))
        intercells[ind] = Q_R[ind]

        #   If left of right shock -> Qc_R_star
        ind = np.where((0 >= U_star) * (P_star > P_R) * (S_star_R >= 0))
        ind_tmp = ind[:3]

        # TODO: to be changed for 2D
        intercells[ind_tmp + (fields.abarrho,)] = abar_R[ind] * rho_R_star[ind]
        intercells[ind_tmp + (fields.rhoU,)] = rho_R_star[ind] * (
            U_R[ind]
            + np.einsum("...k,...l->...kl", U_star - Un_R, normals)[ind_tmp + (0,)]
        )
        intercells[ind + (fields.rhoV,)] = rho_R_star[ind] * (
            V_R[ind]
            + np.einsum("...k,...l->...kl", U_star - Un_R, normals)[ind_tmp + (1,)]
        )
        intercells[ind_tmp + (fields.ad,)] = ad_R_star[ind]
        intercells[ind_tmp + (fields.arho1,)] = arho1_R_star[ind]
        intercells[ind_tmp + (fields.arho2,)] = arho2_R_star[ind]
        intercells[ind_tmp + (fields.arho1d,)] = arho1d_R_star[ind]
        intercells[ind_tmp + (fields.capSigma,)] = capSigma_R_star[ind]

        #   If right fan -> check if in or out of the fan
        index = np.where(-(Un_R - U_star) / c_R / (1 - ad_R) < 100)
        ad_R_star = np.ones_like(ad_R)
        ad_R_star[index] = 1 - 1 / (
            1
            + ad_R[index]
            / (1 - ad_R[index])
            * np.exp(-(Un_R[index] - U_star[index]) / c_R[index] / (1 - ad_R[index]))
        )

        # try:
        #    ad_R_star = 1 - 1 / (
        #        1 + ad_R / (1 - ad_R) * np.exp(-(Un_R - U_star) / c_R / (1 - ad_R))
        #    )
        # except:
        #    for i in range(ad_R_star.shape[0]):
        #        for j in range(ad_R_star.shape[1]):
        #            try:
        #                a = 1 - 1 / (1 + ad_R[i,j,...] / (1 - ad_R[i,j,...]) * np.exp(-(Un_R[i,j,...] - U_star[i,j,...]) / c_R[i,j,...] / (1 - ad_R[i,j,...])))
        #            except:
        #                print(ad_R[i,j,...])
        #                print(Un_R[i,j,...])
        #                print(U_star[i,j,...])
        #                print(c_R[i,j,...])
        #                exit()
        SH_R = Un_R + c_R
        index = np.where((Un_R - U_star) / c_R / (1 - ad_R) < 100)
        ST_R = np.ones_like(U_star) * np.inf
        ST_R[index] = U_star[index] + c_R[index] * (
            1
            + ad_R[index]
            * np.exp((Un_R[index] - U_star[index]) / c_R[index] / (1 - ad_R[index]))
        )
        #       If right of the fan -> Qc_R
        ind = np.where((0 >= U_star) * (P_star <= P_R) * (SH_R < 0))
        ind_tmp = ind[:3]

        intercells[ind] = Q_R[ind]

        #       If in the fan -> Qc_R_fan
        ind = np.where((0 >= U_star) * (P_star <= P_R) * (SH_R >= 0) * (ST_R < 0))
        ind_tmp = ind[:3]

        ad_R_fan = np.ones_like(abar_R) * np.nan
        ad_R_fan[ind] = self.solveAlpha1dFan(
            -Un_R[ind] / c_R[ind] / (1 - ad_R[ind]), ad_R[ind]
        )

        arho1_R_fan = (
            (1 - ad_R_fan)
            * arho1_R
            / (1 - ad_R)
            * np.exp(-(Un_R + (c_R * (1 - ad_R) / (1 - ad_R_fan))) / c_R / (1 - ad_R))
        )
        arho2_R_fan = (
            (1 - ad_R_fan)
            * arho2_R
            / (1 - ad_R)
            * np.exp(-(Un_R + (c_R * (1 - ad_R) / (1 - ad_R_fan))) / c_R / (1 - ad_R))
        )
        arho1d_R_fan = np.where(
            arho1d_R > 0, np.divide(ad_R_fan * arho1d_R, ad_R, where=ad_R > 0), 0
        )
        capSigma_R_fan = np.where(
            capSigma_R > 0, np.divide(ad_R_fan * capSigma_R, ad_R, where=ad_R > 0), 0
        )
        rho_R_fan = arho1_R_fan + arho2_R_fan + arho1d_R_fan

        # TODO: to be changed for 2D
        intercells[ind_tmp + (fields.abarrho,)] = abar_R[ind] * rho_R_fan[ind]
        intercells[ind_tmp + (fields.rhoU,)] = rho_R_fan[ind] * (
            U_R[ind]
            + np.einsum(
                "...k,...l->...kl", -c_R * (1 - ad_R) / (1 - ad_R_fan) - Un_R, normals
            )[ind_tmp + (0,)]
        )
        intercells[ind_tmp + (fields.rhoV,)] = rho_R_fan[ind] * (
            V_R[ind]
            + np.einsum(
                "...k,...l->...kl", -c_R * (1 - ad_R) / (1 - ad_R_fan) - Un_R, normals
            )[ind_tmp + (1,)]
        )
        intercells[ind_tmp + (fields.ad,)] = ad_R_fan[ind]
        intercells[ind_tmp + (fields.arho1,)] = arho1_R_fan[ind]
        intercells[ind_tmp + (fields.arho2,)] = arho2_R_fan[ind]
        intercells[ind_tmp + (fields.arho1d,)] = arho1d_R_fan[ind]
        intercells[ind_tmp + (fields.capSigma,)] = capSigma_R_fan[ind]

        #       If left of the fan -> Qc_R_star
        ind = np.where((0 >= U_star) * (P_star <= P_R) * (ST_R >= 0))
        ind_tmp = ind[:3]

        arho1_R_star = (
            (1 - ad_R_star)
            * arho1_R
            / (1 - ad_R)
            * np.exp(-(Un_R - U_star) / c_R / (1 - ad_R))
        )
        arho2_R_star = (
            (1 - ad_R_star)
            * arho2_R
            / (1 - ad_R)
            * np.exp(-(Un_R - U_star) / c_R / (1 - ad_R))
        )
        arho1d_R_star = np.where(
            arho1d_R > 0, np.divide(ad_R_star * arho1d_R, ad_R, where=ad_R > 0), 0
        )
        capSigma_R_star = np.where(
            capSigma_R > 0, np.divide(ad_R_star * capSigma_R, ad_R, where=ad_R > 0), 0
        )
        rho_R_star = arho1_R_star + arho2_R_star + arho1d_R_star

        intercells[ind_tmp + (fields.abarrho,)] = abar_R[ind] * rho_R_star[ind]
        intercells[ind_tmp + (fields.rhoU,)] = rho_R_star[ind] * (
            U_R[ind]
            + np.einsum("...k,...l->...kl", U_star - Un_R, normals)[ind_tmp + (0,)]
        )
        intercells[ind + (fields.rhoV,)] = rho_R_star[ind] * (
            V_R[ind]
            + np.einsum("...k,...l->...kl", U_star - Un_R, normals)[ind_tmp + (1,)]
        )
        intercells[ind_tmp + (fields.ad,)] = ad_R_star[ind]
        intercells[ind_tmp + (fields.arho1,)] = arho1_R_star[ind]
        intercells[ind_tmp + (fields.arho2,)] = arho2_R_star[ind]
        intercells[ind_tmp + (fields.arho1d,)] = arho1d_R_star[ind]
        intercells[ind_tmp + (fields.capSigma,)] = capSigma_R_star[ind]

        return intercells

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
        intercells = Q_L.copy()

        # Test if discontinuity
        ind = np.where(
            np.any(
                Q_L.view(Q).get_conservative() != Q_R.view(Q).get_conservative(),
                axis=-1,
            )
        )
        if ind[0].size > 0:
            intercells[ind] = self.solve_RP(Q_L, Q_R, intercells, normals)[ind]
        # Compute flux
        self.auxilliaryVariableUpdateNoGeo(intercells)
        F = np.einsum("...mkl,...l->...mk", self.problem.F_hyper(intercells), normals)

        # Multiply by surfaces
        FS.set_conservative(surfaces[..., np.newaxis, np.newaxis] * F)

        return FS
