# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from josie.mesh.cellset import MeshCellSet
from josie.scheme.convective import ConvectiveScheme
from josie.twofluid.state import PhasePair
from josie.twofluid.fields import Phases

from .eos import TwoPhaseEOS
from .problem import TsCapProblem
from .state import Q, TsCapPhaseFields

from josie.mesh.cellset import NeighbourDirection


class TsCapScheme(ConvectiveScheme):
    """A base class for a twophase scheme with capillarity"""

    problem: TsCapProblem
    dx: float
    dy: float
    tmp_arr: np.ndarray
    nSmoothPass: int
    noMassTransfer: bool

    def __init__(
        self,
        eos: TwoPhaseEOS,
        sigma: float,
        Hmax: float,
        kappa: float,
        dx: float,
        dy: float,
        norm_grada_min: float,
        nSmoothPass: int,
    ):
        super().__init__(TsCapProblem(eos, sigma, Hmax, kappa, norm_grada_min))

        self.geoUpdate = True

        self.dx = dx
        self.dy = dy
        self.nSmoothPass = nSmoothPass
        dim = 2
        if dim > 0:
            self.directions = [
                NeighbourDirection.LEFT,
                NeighbourDirection.RIGHT,
            ]

        if dim > 1:
            self.directions.extend((NeighbourDirection.BOTTOM, NeighbourDirection.TOP))
        self.noMassTransfer = False

    def post_init(self, cells):
        super().post_init(cells)

        # self.mean = np.zeros_like(cells.values[..., 0])
        # self.cnt = np.zeros_like(cells.values[..., 0])
        # self.std = np.zeros_like(cells.values[..., 0])
        self.phi_out = np.zeros_like(cells.values[..., 0])

    def nan_gradient(self, data, dx, dy):
        """Compute the gradient of a 2D array in the 2 directions, 2 nd order
        in the interior of the non-nan object, 1 st order at the interface between
        the non-nan object and the surrounding nan values.

        :param data: the 2D array to be derived (2D np.ndarray)
        :param dx: the spacing in the x direction (axis 0)
        :param dy: the spacing in the y direction (axis 1)

        :return: a tuple, the two gradients (in each direction) with the
        same shape as the input data
        """

        grad_x = (data[1:, ...] - data[:-1, ...]) / dx
        grad_y = (data[:, 1:] - data[:, :-1]) / dy

        out_grad_x = np.full_like(grad_x[1:, 1:-1, ...], np.nan)
        out_grad_y = np.full_like(grad_y[1:-1, 1:], np.nan)
        # Calcul 1
        index = np.where(
            np.invert(
                np.all(
                    np.isnan(np.stack([grad_x[1:, 1:-1, ...], grad_x[:-1, 1:-1, ...]])),
                    axis=0,
                )
            )
        )
        out_grad_x[index] = np.nanmean(
            np.stack([grad_x[1:, 1:-1, ...][index], grad_x[:-1, 1:-1, ...][index]]),
            axis=0,
        )
        index = np.invert(
            np.all(np.isnan(np.stack([grad_y[1:-1, 1:], grad_y[1:-1, :-1]])), axis=0)
        )
        out_grad_y[index] = np.nanmean(
            np.stack([grad_y[1:-1, 1:][index], grad_y[1:-1, :-1][index]]),
            axis=0,
        )
        if len(out_grad_x.shape) == 2:
            return (
                np.pad(out_grad_x, ((1, 1), (1, 1)), constant_values=np.nan),
                np.pad(out_grad_y, ((1, 1), (1, 1)), constant_values=np.nan),
            )
        elif len(out_grad_x.shape) == 3:
            return (
                np.pad(out_grad_x, ((1, 1), (1, 1), (0, 0)), constant_values=np.nan),
                np.pad(out_grad_y, ((1, 1), (1, 1), (0, 0)), constant_values=np.nan),
            )
        else:
            exit()

    def remove_outlier(self, field, n):
        self.mask = np.zeros_like(field[1:-1, 1:-1])
        self.cnt = np.zeros_like(field[1:-1, 1:-1])
        self.mean = np.zeros_like(field[1:-1, 1:-1])
        self.std = np.zeros_like(field[1:-1, 1:-1])
        if n == 0:
            return field
        else:
            self.tmp_arr.fill(np.nan)
            # Compute mean over the patch of neigbours
            # for i, dir in enumerate(self.directions):
            #     self.tmp_arr[..., i + 1] = field[dir.value.data_index]

            # field[1:-1, 1:-1] = np.where(
            #     (~np.isnan(field[1:-1, 1:-1])),
            #     np.where(
            #         np.abs(field[1:-1, 1:-1] - np.nanmean(self.tmp_arr, axis=-1))
            #         > np.nanstd(self.tmp_arr, axis=-1),
            #         np.nanmean(self.tmp_arr, axis=-1),
            #         field[1:-1, 1:-1],
            #     ),
            #     np.nan,
            # )
            self.cnt.fill(0)
            self.mean.fill(0)
            self.std.fill(0)
            for dir in self.directions:
                self.mask = ~np.isnan(field[dir.value.data_index])
                self.cnt += ~np.isnan(field[dir.value.data_index])
                self.mean += np.where(self.mask, field[dir.value.data_index], 0)
            self.mean = np.divide(self.mean, self.cnt, where=self.cnt > 0)
            for dir in self.directions:
                self.mask = ~np.isnan(field[dir.value.data_index])
                self.std += np.where(
                    self.mask, (field[dir.value.data_index] - self.mean) ** 2, 0
                )
            self.std = np.sqrt(self.std, where=self.std > 0)

            field[1:-1, 1:-1] = np.where(
                (~np.isnan(field[1:-1, 1:-1])),
                np.where(
                    np.abs(field[1:-1, 1:-1] - self.mean) > self.std,
                    self.mean,
                    field[1:-1, 1:-1],
                ),
                np.nan,
            )
            #     self.std[self.mask] += (
            #         field[dir.value.data_index][self.mask] - self.mean[self.mask]
            #     ) ** 2
            # self.std = np.sqrt(self.std, where=self.std > 0)

            # self.mask = ~np.isnan(field[1:-1, 1:-1])
            # field[1:-1, 1:-1][self.mask] = np.where(
            #     np.abs(field[1:-1, 1:-1] - self.mean) > self.std,
            #     self.mean,
            #     field[1:-1, 1:-1],
            # )[self.mask]

            # Recursive call
            return self.remove_outlier(field, n - 1)

    def relaxation(self, values: Q):
        fields = Q.fields
        #if len(values.shape)<4:
        #    exit()
        arho1 = values[..., fields.arho1]
        arho2 = values[..., fields.arho2]
        arho1d = values[..., fields.arho1d]
        ad = values[..., fields.ad]
        rho1d = np.full_like(arho1d, self.problem.eos[Phases.PHASE1].rho0)
        np.divide(arho1d, ad, where=(arho1d > 0) & (ad > 0), out=rho1d)
        capSigma = values[..., fields.capSigma]
        abarrho = values[..., fields.abarrho]
        rhoU = values[..., fields.rhoU]
        rhoV = values[..., fields.rhoV]
        rho = arho1 + arho2 + arho1d
        if np.any(abarrho<0):
            if np.min(abarrho)<-1e-10:
                print("Negative abarrho at beginning of relaxation")
                print(abarrho[np.where(abarrho<0)])
            abarrho = np.maximum(abarrho,0)

        if np.any(arho1<0):
            if np.nanmin(arho1) < -1e-14: 
                print(np.nanmin(arho1))
                index = np.where(arho1==np.nanmin(arho1))
                print(abarrho[index])
                print(arho1d[index])
                print(arho2[index])
                exit()
            arho1 = np.maximum(arho1,0)
        if np.any(ad>1):
            print(np.max(ad))
            exit()
        if np.any(arho2<0):
            if np.min(arho2) < -1e-14:
                print(np.min(arho2))
                exit()
            arho2 = np.maximum(arho2, 0)
            

        self.a1dc = 0.5
        # Compute estimator of the relaxation within [0,1]
        abar = np.minimum(np.maximum(abarrho / rho, 0), 1)

        rho1 = np.full_like(arho1, np.nan)
        rho2 = np.full_like(arho1, np.nan)
        np.divide(arho1, abar * (1 - ad), where=(abar > 0) & (arho1 > 0), out=rho1)
        np.divide(
            arho2,
            (1 - abar) * (1 - ad),
            where=((1.0 - abar) > 0) & (arho2 > 0),
            out=rho2,
        )
        if np.any(rho1<0):
            exit()
        values[..., fields.abar] = abar
        if self.geoUpdate:
            self.updateGeometry(values)
        H = values[..., fields.H]
        grada_x = values[..., fields.grada_x]
        grada_y = values[..., fields.grada_y]
        Hlim = H.copy()
        if not (self.noMassTransfer):
            # Mass transfer location conditions
            Hlim = np.where(
                (3 / self.problem.kappa / rho1d * rho1 * (1 - ad) - (1 - abar) > 0)
                & (abar > 0.01)
                & (abar < 0.1)
                & (-grada_x * rhoU - grada_y * rhoV > 0)
                & (ad < self.a1dc),
                np.minimum(H, self.problem.Hmax),
                Hlim,
            )
        DH = H - Hlim

        # Note that rho remain constant
        def phi(
            ad: np.ndarray,
            Hlim: np.ndarray,
            rho1: np.ndarray,
            rho2: np.ndarray,
        ):
            return (1 - ad) * (
                self.problem.eos[Phases.PHASE1].p(rho1)
                - self.problem.eos[Phases.PHASE2].p(rho2)
            ) - self.problem.sigma * Hlim

        def dphi_dabar(
            arho1: np.ndarray,
            arho2: np.ndarray,
            abar: np.ndarray,
            rho1: np.ndarray,
            rho2: np.ndarray,
        ):
            # Note that dp_drho = c^2 for barotropic EOS
            try:
                return (
                    -arho1
                    / (abar**2)
                    * self.problem.eos[Phases.PHASE1].sound_velocity(rho1) ** 2
                    - arho2
                    / ((1.0 - abar) ** 2)
                    * self.problem.eos[Phases.PHASE2].sound_velocity(rho2) ** 2
                )
            except:
                abar = np.ravel(abar)
                for x in abar:
                    try:
                        y = 1 / (x**2)
                        y = 1 / ((1.0-x)**2)
                    except:
                        print(x)
                        exit()

        def dphi_dm1(rho1: np.ndarray, abar: np.ndarray):
            return self.problem.eos[Phases.PHASE1].sound_velocity(rho1) ** 2 / abar

        def dphi_dad(
            rho1: np.ndarray,
            rho2: np.ndarray,
        ):
            p1 = self.problem.eos[Phases.PHASE1].p(rho1)
            c1 = self.problem.eos[Phases.PHASE1].sound_velocity(rho1)
            p2 = self.problem.eos[Phases.PHASE2].p(rho2)
            c2 = self.problem.eos[Phases.PHASE2].sound_velocity(rho2)
            return p2 - p1 + c1**2 * rho1 - c2**2 * rho2

        def compute_step(
            dc: float,
            arho1: np.ndarray,
            abar: np.ndarray,
            ad: np.ndarray,
            DH: np.ndarray,
            R: np.ndarray,
            F: np.ndarray,
            dFda: np.ndarray,
            rho1: np.ndarray,
            momDotVel: np.ndarray,
            fac: np.ndarray,
            rho1d: np.ndarray,
        ) -> np.ndarray:
            # arho1 stability condition
            dtau = np.full_like(arho1, np.inf)
            np.divide(
                dc * arho1,
                (rho1 * self.problem.sigma / (1 - abar) * DH),
                where=(DH > 0) & (~np.isnan(rho1)),
                out=dtau,
            )
            if np.any(dtau < 0):
                print(rho1[np.where(rho1<0)])
                exit()

            # velocity stability condition
            dtautmp = np.full_like(dtau, np.inf)
            np.divide(
                momDotVel,
                self.problem.Hmax * DH * fac * self.problem.sigma**2,
                where=(DH > 0) & (~np.isnan(rho1)) & (fac >0),
                out=dtautmp,
            )
            dtau = np.minimum(
                dtau,
                dtautmp,
            )
            if np.any(dtau < 0):
                exit()
            # ad stability condition
            dtautmp = np.full_like(ad, np.inf)
            np.divide(
                dc * (self.a1dc - ad),
                (rho1 * self.problem.sigma / (1 - abar) / rho1d * DH),
                where=(DH > 0) & (~np.isnan(rho1)),
                out=dtautmp,
            )
            dtau = np.minimum(
                dtau,
                dtautmp,
            )
            if np.any(dtau < 0):
                exit()
            dtautmp = np.full_like(ad, np.inf)
            np.divide(
                ad,
                (rho1 * self.problem.sigma / (1 - abar) / rho1d * DH),
                where=(DH > 0) & (~np.isnan(rho1)) & (ad < self.a1dc) & (ad > 0),
                out=dtautmp,
            )
            dtau = np.minimum(
                dtau,
                dtautmp,
            )
            if np.any(dtau < 0):
                exit()

            # abar stability condition
            a = rho1 * self.problem.sigma / (1 - abar) / (1 - ad) * DH * R
            # Upper bound
            b = 1 / (1 - ad) * (F + dc * (1 - abar) * dFda)
            try:
                D = b**2 - 4 * a * (-dc * (1 - abar))
            except:
                b = np.ravel(b)
                for x in b:
                    try:
                        res = x**2
                    except:
                        print(x)
                        exit()
            # This gives the first root when a<0
            # or the second when a>0
            dtautmp = np.full_like(a, np.inf)
            np.divide(
                (-b + np.sqrt(D, where=D > 0)),
                (2 * a),
                where=(D > 0) & ((a > 0) | ((a < 0) & (b > 0))),
                out=dtautmp,
            )
            dtautmp = np.where((a == 0) & (b > 0), dc * (1 - abar) / b, dtautmp)
            # D = np.zeros_like(a)
            # indtmp = np.where((DH > 0))
            # D[indtmp] = np.power(b[indtmp], 2) - 4 * a[indtmp] * (
            #     -dc * (1 - abar[indtmp])
            # )
            # # This gives the first root when a<0
            # # or the second when a>0
            # dtautmp = np.full_like(a, np.inf)
            # indtmp = np.where((D > 0) & ((a > 0) | ((a < 0) & (b > 0))) & (DH > 0))
            # dtautmp[indtmp] = (-b[indtmp] + np.sqrt(D[indtmp])) / (2 * a[indtmp])
            # indtmp = np.where((a == 0) & (b > 0) & (DH > 0))
            # dtautmp[indtmp] = dc * (1 - abar[indtmp])
            dtau = np.minimum(
                dtau,
                dtautmp,
            )
            if np.any(dtau < 0):
                exit()

            # Lower bound
            #print(abar[117,95])
            b = 1 / (1 - ad) * (F - dc * abar * dFda)
            D = b**2 - 4 * a * (dc * abar)
            dtautmp = np.full_like(a, np.inf)
            np.divide(
                (-b - np.sqrt(D, where=D > 0)),
                (2 * a),
                where=(D > 0) & ((a < 0) | ((a > 0) & (b < 0))),
                out=dtautmp,
            )
            dtautmp = np.where((a == 0) & (b < 0), -(dc * abar) / b, dtautmp)
            # D.fill(0)
            # indtmp = np.where((DH > 0))
            # D[indtmp] = np.power(b[indtmp], 2) - 4 * a[indtmp] * (dc * abar[indtmp])
            # dtautmp.fill(np.inf)
            # indtmp = np.where((D > 0) & ((a < 0) | ((a > 0) & (b < 0))) & (DH > 0))
            # dtautmp[indtmp] = (-b[indtmp] - np.sqrt(D[indtmp])) / (2 * a[indtmp])
            # indtmp = np.where((a == 0) & (b < 0) & (DH > 0))
            # dtautmp[indtmp] = -(dc * abar[indtmp]) / b[indtmp]
            dtau = np.minimum(
                dtau,
                dtautmp,
            )
            if np.any(dtau < 0):
                exit()
            #print("end_dtau")
            return dtau

        # Init NR method
        dabar = np.zeros_like(abar)
        iter = 0

        # Index that locates the cell where there the pressures need to be relaxed
        eps = 1e-20
        tol = 1e-10
        p0 = self.problem.eos[Phases.PHASE1].p0
        p0 = np.minimum(
            np.abs(self.problem.sigma * Hlim), self.problem.eos[Phases.PHASE1].p0
        )
        F = phi(ad, Hlim, rho1, rho2)
        index = np.where((abar < eps))
        arho1[index] = abar[index] * self.problem.eos[Phases.PHASE1].rho0
        index = np.where((1-abar < eps))
        arho2[index] = (1-abar[index]) * self.problem.eos[Phases.PHASE2].rho0

        index = np.where((np.abs(F) > tol * p0) & (abar > eps) & (1 - abar > eps))
        self.fac = np.zeros_like(abar)
        self.drho_fac = np.zeros_like(abar)
        while index[0].size > 0:
            # Counter
            iter += 1
            

            # Damping coef
            dc = 0.9

            R = dphi_dad(rho1[index], rho2[index]) / rho1d[index] - dphi_dm1(
                rho1[index], abar[index]
            )
            momDotVel = (rhoU**2 + rhoV**2) / rho
            self.fac[index] = np.where(
                (DH > 0)[index],
                np.maximum(0, 3 / self.problem.kappa / rho1d[index] * rho1[index] / (1 - abar[index])
                - 1 / (1 - ad[index])),
                0,
            )
            # fac = 3 * rho1[index] / (
            #     self.problem.kappa * rho1d[index] * (1 - abar[index])
            # ) - 1 / (1 - ad[index])
            # dtau ~ dtau / epsilon
            if np.any(abar[index]==0) or np.any(abar[index]==1):
                exit()
            dFda = dphi_dabar(
                arho1[index], arho2[index], abar[index], rho1[index], rho2[index]
            )
            dtau = compute_step(
                dc,
                arho1[index],
                abar[index],
                ad[index],
                DH[index],
                R,
                F[index],
                dFda,
                rho1[index],
                momDotVel[index],
                self.fac[index],
                # fac,
                rho1d[index],
            )
            if np.any(dtau < 0) or np.any(np.isnan(dtau)):
                raise Exception("Negative timestep in relaxation: " + str(dtau))

            darho1 = np.zeros_like(dtau)
            np.multiply(
                -dtau,
                arho1[index]
                / abar[index]
                / (1 - ad[index])
                / (1 - abar[index])
                * self.problem.sigma
                * DH[index],
                where=~np.isinf(dtau),
                out=darho1,
            )
            dabar = np.zeros_like(abar)
            try:
                dabar[index] = np.where(
                    np.isinf(dtau),
                    -F[index] / dFda,
                    (
                        np.divide(
                            dtau / (1 - ad[index]) * (F[index] - darho1 * R),
                            (1 - dtau / (1 - ad[index]) * dFda),
                            where=~np.isinf(dtau),
                        )
                    ),
                )
            except:
                ad = np.ravel(ad[index])
                F = np.ravel(F[index])
                darho1 = np.ravel(darho1)
                R = np.ravel(R)
                dFda = np.ravel(dFda)
                dtau = np.ravel(dtau)
                for i in range(len(ad)):
                    try:
                        y = dtau / (1-ad[i]) * (F[i]-darho1[i]*R[i])
                    except:
                        for j in range(100):
                            print("data")
                            print(dtau[i])
                            print(ad[i])
                            print(F[i])
                            print(darho1[i])
                            print(R[i])
                        exit()
                exit()
            if np.any(DH[index] > 0):
                self.drho_fac.fill(0)
                self.drho_fac[index] = np.multiply(
                    dtau,
                    np.divide(
                        self.problem.sigma**2
                        * DH[index]
                        * self.fac[index]
                        * Hlim[index]
                        * rho[index],
                        (rhoU**2 + rhoV**2)[index],
                        where=(rhoU**2 + rhoV**2)[index] > 0,
                    ),
                    where=DH[index] > 0,
                )
                #     np.divide(
                #         dtau / (1 - ad[index]) * (F[index] - darho1 * R),
                #         (1 - dtau / (1 - ad[index]) * dFda),
                #         where=~np.isinf(dtau),
                #     ),
                # )
                # indtmp = np.where(~np.isinf(dtau))
                # drho_fac = np.zeros_like(dtau)
                # drho_fac[indtmp] = (
                #     dtau[indtmp]
                #     * self.problem.sigma**2
                #     * DH[index][indtmp]
                #     * fac[indtmp]
                #     * Hlim[index][indtmp]
                #     * rho[index][indtmp]
                #     / (rhoU**2 + rhoV**2)[index][indtmp]
                # )

                drhoU = -self.drho_fac[index] * rhoU[index]
                drhoV = -self.drho_fac[index] * rhoV[index]
            else:
                drhoU = 0 * rhoU[index]
                drhoV = 0 * rhoV[index]

            # Update values
            #dabar[np.where(np.abs(dabar)<1e-15)] = 0
            if np.any(ad[index] - darho1 / rho1d[index]>1):
                index2 = np.where(ad[index] - darho1 / rho1d[index]>1)
                exit()
            if np.any(abar[index]<0):
                exit()
            if np.any(abar[index] + dabar[index]<0):
                index2 = np.where(abar[index] + dabar[index]<0)
                print(
                    dc,
                    arho1[index][index2],
                    abar[index][index2],
                    ad[index][index2],
                    DH[index][index2],
                    R[index2],
                    F[index][index2],
                    dFda[index2],
                    rho1[index][index2],
                    momDotVel[index][index2],
                    self.fac[index][index2],
                    rho1d[index][index2],
                )
                print(dtau[index2])
                print(dabar[index][index2])
                a = rho1[index][index2] * self.problem.sigma / (1 - abar[index][index2]) / (1 - ad[index][index2]) * DH[index][index2] * R[index2]
                b = 1 / (1 - ad[index][index2]) * (F[index][index2] - dc * abar[index][index2] * dFda[index2])
                print(a)
                print(b)
                print(np.where((a == 0) & (b < 0), -(dc * abar[index][index2]) / b, np.inf))
                print(np.where(abar==abar[index][index2][0]))
                exit()
            abar[index] += dabar[index]

            if np.any(abar[index]<0):
                #print(abar[index][np.where(abar[index]<0)])
                #index2 = np.where(abar[index]<0)
                #print(dtau[index2])
                if np.any(abar<0):
                    print(index)
                    if np.all(abar[np.where(abar<0)]>-1e-16):
                        print(abar[np.where(abar<0)])
                        abar = np.maximum(abar,0)
                    else:
                        print(np.min(abar[np.where(abar<0)]))
                        exit()
                else:
                    print(np.nanmin(abar))
                    print(abar.shape)
                    exit()
            arho1[index] += darho1
            if np.any(arho1<0):
                exit()
            capSigma[index] -= (
                3 * self.problem.Hmax / self.problem.kappa / rho1d[index]
            ) * darho1
            rhoU[index] += drhoU
            rhoV[index] += drhoV
            if np.any(darho1>0):
                print(np.nanmin(dtau))
                print(np.nanmax(ad[index]))
                print(np.nanmax(abar[index]))
                print(np.nanmin(DH[index]))
                exit()
            arho1d[index] -= darho1
            ad[index] -= darho1 / rho1d[index]
            values[..., fields.abarrho] = abar * rho

            # Update the index where the NR method is applied
            if np.any(DH > 1e-7):
                self.updateGeometry(values)
                H = values[..., fields.H]
                Hlim = H.copy()
                if not (self.noMassTransfer):
                    try:
                        g = -grada_x * rhoU - grada_y * rhoV
                    except:
                        grada_x = np.ravel(grada_x)
                        rhoU = np.ravel(rhoU)
                        grada_y = np.ravel(grada_y)
                        rhoV = np.ravel(rhoV)
                        for i in range(len(grada_x)):
                            try:
                                g = -grada_x * rhoU - grada_y * rhoV
                            except:
                                print(grada_x[i], rhoU[i], rhoV[i], grada_y[i])
                                exit()
                    Hlim = np.where(
                        (
                            3 / self.problem.kappa / rho1d * rho1 * (1 - ad)
                            - (1 - abar)
                            > 0
                        )
                        & (abar > 0.01)
                        & (abar < 0.1)
                        & (-grada_x * rhoU - grada_y * rhoV > 0)
                        & (ad < self.a1dc),
                        np.minimum(H, self.problem.Hmax),
                        Hlim,
                    )
                DH = H - Hlim
                p0 = np.minimum(
                    np.abs(self.problem.sigma * Hlim), self.problem.eos[Phases.PHASE1].p0
                )
            rho1.fill(np.nan)
            rho2.fill(np.nan)
            np.divide(arho1, abar * (1 - ad), where=(abar > 0) & (arho1 > 0), out=rho1)
            np.divide(
                arho2,
                (1 - abar) * (1 - ad),
                where=((1.0 - abar) > 0) & (arho2 > 0),
                out=rho2,
            )
            F = phi(ad, Hlim, rho1, rho2)
            index = np.where((abar < eps))
            arho1[index] = abar[index] * self.problem.eos[Phases.PHASE1].rho0
            index = np.where((1-abar < eps))
            arho2[index] = (1-abar[index]) * self.problem.eos[Phases.PHASE2].rho0
            index = np.where(
                (np.abs(F) > tol * p0)
                & (abar > eps)
                & (1 - abar > eps)
                & (np.abs(dabar) > tol)
            )

            # Safety check
            if iter > 30:
                self.noMassTransfer = True

            if iter > 60:
                raise Exception(
                    "Maximal iteration reached in relaxation with |dabar|: "
                    + str(np.abs(dabar))
                )

        if np.any(ad>1):
            exit()
        if np.any(arho1<0):
            exit()
        if np.any(arho1d<0):
            print(np.nanmin(arho1d))
            exit()
        if np.any(abar<0):
            exit()
        if np.any(arho2<0):
            exit()
        if np.any(abar>1):
            index = np.where(abar>1)
            print(abar[index])
            exit()
        self.noMassTransfer = False

        values[..., fields.arho1] = arho1
        values[..., fields.arho1d] = arho1d
        values[..., fields.arho2] = arho2
        values[..., fields.abarrho] = abar * (arho1 + arho2 + arho1d)
        values[..., fields.ad] = ad
        values[..., fields.capSigma] = capSigma
        values[..., fields.rhoU] = rhoU
        values[..., fields.rhoV] = rhoV

    def updateGeometry(self, values: Q):
        fields = Q.fields
        dx = self.dx
        dy = self.dy

        # Geometric periodic update
        periodic = False
        abar = values[..., fields.abar]
        if periodic:
            # Left
            abar[0, ...] = abar[-2, ...]
            # Right
            abar[-1, ...] = abar[1, ...]
            # Top
            abar[:, 0, ...] = abar[:, -2, ...]
            # Bottom
            abar[:, -1, ...] = abar[:, 1, ...]
        else:
            # Left
            abar[0, ...] = abar[1, ...]
            # Right
            abar[-1, ...] = abar[-2, ...]
            # Top
            abar[:, 0, ...] = abar[:, 1, ...]
            # Bottom
            abar[:, -1, ...] = abar[:, -2, ...]
        grada_x, grada_y = np.gradient(abar, dx, dy, axis=(0, 1))

        if periodic:
            # Geometric periodic update
            # Left
            grada_x[0, ...] = grada_x[-2, ...]
            grada_y[0, ...] = grada_y[-2, ...]
            # Right
            grada_x[-1, ...] = grada_x[1, ...]
            grada_y[-1, ...] = grada_y[1, ...]
            # Top
            grada_x[:, 0, ...] = grada_x[:, -2, ...]
            grada_y[:, 0, ...] = grada_y[:, -2, ...]
            # Bottom
            grada_x[:, -1, ...] = grada_x[:, 1, ...]
            grada_y[:, -1, ...] = grada_y[:, 1, ...]
        else:
            # Left
            grada_x[0, ...] = grada_x[1, ...]
            grada_y[0, ...] = grada_y[1, ...]
            # Right
            grada_x[-1, ...] = grada_x[-2, ...]
            grada_y[-1, ...] = grada_y[-2, ...]
            # Top
            grada_x[:, 0, ...] = grada_x[:, 1, ...]
            grada_y[:, 0, ...] = grada_y[:, 1, ...]
            # Bottom
            grada_x[:, -1, ...] = grada_x[:, -2, ...]
            grada_y[:, -1, ...] = grada_y[:, -2, ...]

        norm_grada = np.sqrt(grada_x**2 + grada_y**2)
        n_x = np.full_like(grada_x, np.nan)
        n_y = np.full_like(grada_y, np.nan)
        np.divide(
            grada_x,
            norm_grada,
            where=(norm_grada > 0) & (abar > 0) & (abar < 1),
            out=n_x,
        )
        np.divide(
            grada_y,
            norm_grada,
            where=(norm_grada > 0) & (abar > 0) & (abar < 1),
            out=n_y,
        )

        H = -(self.nan_gradient(n_x, dx, dy)[0] + self.nan_gradient(n_y, dx, dy)[1])
        H[np.where(np.abs(H) < 1e-10)] = 0

        # Smoothening
        if len(H.shape) == 3:
            H[..., 0] = self.remove_outlier(H[..., 0], self.nSmoothPass)
        elif len(H.shape) == 2:
            H = self.remove_outlier(H, self.nSmoothPass)

        values[..., fields.grada_x] = grada_x
        values[..., fields.grada_y] = grada_y
        values[..., fields.n_x] = n_x
        values[..., fields.n_y] = n_y
        values[..., fields.norm_grada] = norm_grada
        values[..., fields.H] = H

    def auxilliaryVariableUpdate(self, values: Q):
        fields = Q.fields
        # sigma = self.problem.sigma

        # Get variables updated by the scheme
        abarrho = values[..., fields.abarrho]
        arho1 = values[..., fields.arho1]
        arho2 = values[..., fields.arho2]
        arho1d = values[..., fields.arho1d]
        rhoU = values[..., fields.rhoU]
        rhoV = values[..., fields.rhoV]
        ad = values[..., fields.ad]
        
        if np.any(arho1<0):
            print(np.nanmin(arho1))
            exit()
        if np.any(arho1d<0):
            print(np.nanmin(arho1d))
            exit()
        if np.any(arho2<0):
            print(np.nanmin(arho2))
            exit()

        # Updating the auxiliary variables
        rho = arho1 + arho2 + arho1d
        abar = abarrho / rho
        U = rhoU / rho
        V = rhoV / rho

        abars = PhasePair(abar, 1.0 - abar)
        arhos = PhasePair(arho1, arho2)

        values[..., fields.abar] = abar
        values[..., fields.rho] = rho
        values[..., fields.U] = U
        values[..., fields.V] = V

        self.updateGeometry(values)

        c_sq = np.zeros_like(arho1)  # Auxiliary variable for mixture speed of sound

        for phase in Phases:
            phase_values = values.view(Q).get_phase(phase)

            abar = abars[phase]
            arho = arhos[phase]

            rho = np.full_like(arho, np.nan)
            np.divide(arho, abar * (1 - ad), where=(abar > 0), out=rho)
            p = self.problem.eos[phase].p(rho)
            c = self.problem.eos[phase].sound_velocity(rho)

            phase_values[..., TsCapPhaseFields.p] = p
            phase_values[..., TsCapPhaseFields.c] = c

            values.view(Q).set_phase(
                phase,
                phase_values,
            )

            c_sq += np.where(arho > 0, arho * c**2, 0)

        rho = values[..., fields.rho]
        if np.any(c_sq<0):
            exit()
        if np.any(rho<0):
            index = np.where(rho<0)

            print(rho[index][0])
            print(arho1[index][0])
            print(arho1d[index][0])
            print(arho2[index][0])
            print(abar[index][0])
            print(ad[index][0])
            exit()
        cFd = np.sqrt(c_sq / rho) / (1 - ad)

        # MaX = U / cFd
        # MaY = V / cFd
        # norm_grada = values[..., fields.norm_grada]
        # n_x = values[..., fields.n_x]
        # n_y = values[..., fields.n_y]
        # WeX = np.empty_like(norm_grada)
        # WeX[:] = np.nan
        # np.divide(rho * U**2, sigma * norm_grada, where=norm_grada > 0, out=WeX)
        # WeY = np.empty_like(norm_grada)
        # WeY[:] = np.nan
        # np.divide(rho * V**2, sigma * norm_grada, where=norm_grada > 0, out=WeY)
        # r = np.empty_like(norm_grada)
        # r[:] = np.nan
        # np.divide(MaX**2, WeX, where=WeX > 0, out=r)
        # c_cap1X = cFd * (1 + 0.5 * r * n_x**2 * (1 - n_x**2))
        # c_cap2X = cFd * (1 - n_x**2) * np.sqrt(r)
        # r[:] = np.nan
        # np.divide(MaY**2, WeY, where=WeY > 0, out=r)
        # c_cap1Y = cFd * (1 + 0.5 * r * n_y**2 * (1 - n_y**2))
        # c_cap2Y = cFd * (1 - n_y**2) * np.sqrt(r)

        # Update the auxilliary variables
        abar = values[..., fields.abar]
        p1 = values[..., fields.p1]
        p2 = values[..., fields.p2]
        values[..., fields.pbar] = np.nan
        values[..., fields.pbar] = np.where(
            (abar > 0) * (abar < 1),
            abar * p1 + (1 - abar) * p2,
            np.where(abar == 0, p2, p1),
        )

        values[..., fields.cFd] = cFd
        # values[..., fields.MaX] = MaX
        # values[..., fields.MaY] = MaY
        # values[..., fields.WeX] = WeX
        # values[..., fields.WeY] = WeY
        # values[..., fields.c_cap1X] = c_cap1X
        # values[..., fields.c_cap1Y] = c_cap1Y
        # values[..., fields.c_cap2X] = c_cap2X
        # values[..., fields.c_cap2Y] = c_cap2Y

    def auxilliaryVariableUpdateNoGeo(self, values: Q):
        fields = Q.fields
        # sigma = self.problem.sigma

        # Get variables updated by the scheme
        abarrho = values[..., fields.abarrho]
        arho1 = values[..., fields.arho1]
        arho2 = values[..., fields.arho2]
        arho1d = values[..., fields.arho1d]
        rhoU = values[..., fields.rhoU]
        rhoV = values[..., fields.rhoV]
        ad = values[..., fields.ad]

        # Updating the auxiliary variables
        rho = arho1 + arho2 + arho1d
        abar = abarrho / rho
        U = rhoU / rho
        V = rhoV / rho

        abars = PhasePair(abar, 1.0 - abar)
        arhos = PhasePair(arho1, arho2)

        values[..., fields.abar] = abar
        values[..., fields.rho] = rho
        values[..., fields.U] = U
        values[..., fields.V] = V

        c_sq = np.zeros_like(arho1)  # Auxiliary variable for mixture speed of sound

        for phase in Phases:
            phase_values = values.view(Q).get_phase(phase)

            abar = abars[phase]
            arho = arhos[phase]

            rho = np.full_like(arho, np.nan)
            np.divide(arho, abar * (1 - ad), where=((abar > 0) & (ad < 1)), out=rho)
            p = self.problem.eos[phase].p(rho)
            c = self.problem.eos[phase].sound_velocity(rho)

            phase_values[..., TsCapPhaseFields.p] = p
            phase_values[..., TsCapPhaseFields.c] = c

            values.view(Q).set_phase(
                phase,
                phase_values,
            )

            c_sq += np.where(arho > 0, arho * c**2, 0)

        rho = values[..., fields.rho]
        cFd = np.full_like(rho, np.nan)
        np.divide(np.sqrt(c_sq / rho), (1 - ad), where=(ad < 1), out=cFd)

        # MaX = U / cFd
        # MaY = V / cFd
        # norm_grada = values[..., fields.norm_grada]
        # n_x = values[..., fields.n_x]
        # n_y = values[..., fields.n_y]
        # WeX = np.empty_like(norm_grada)
        # WeX[:] = np.nan
        # np.divide(sigma * norm_grada, rho * U**2, where=U**2 > 0, out=WeX)
        # WeY = np.empty_like(norm_grada)
        # WeY[:] = np.nan
        # np.divide(sigma * norm_grada, rho * V**2, where=V**2 > 0, out=WeY)
        # r = np.empty_like(norm_grada)
        # r[:] = np.nan
        # np.divide(MaX**2, WeX, where=WeX > 0, out=r)
        # c_cap1X = cFd * (1 + 0.5 * r * n_x**2 * (1 - n_x**2))
        # c_cap2X = cFd * (1 - n_x**2) * np.sqrt(r)
        # r[:] = np.nan
        # np.divide(MaY**2, WeY, where=WeY > 0, out=r)
        # c_cap1Y = cFd * (1 + 0.5 * r * n_y**2 * (1 - n_y**2))
        # c_cap2Y = cFd * (1 - n_y**2) * np.sqrt(r)

        # Update the auxilliary variables
        abar = values[..., fields.abar]
        p1 = values[..., fields.p1]
        p2 = values[..., fields.p2]
        values[..., fields.pbar] = np.nan
        values[..., fields.pbar] = np.where(
            (abar > 0) * (abar < 1),
            abar * p1 + (1 - abar) * p2,
            np.where(abar == 0, p2, p1),
        )

        values[..., fields.cFd] = cFd
        # values[..., fields.MaX] = MaX
        # values[..., fields.MaY] = MaY
        # values[..., fields.WeX] = WeX
        # values[..., fields.WeY] = WeY
        # values[..., fields.c_cap1X] = c_cap1X
        # values[..., fields.c_cap1Y] = c_cap1Y
        # values[..., fields.c_cap2X] = c_cap2X
        # values[..., fields.c_cap2Y] = c_cap2Y

    def post_step(self, values: Q):
        """During the step we update the conservative values. After the
        step we update the non-conservative variables. This method updates
        the values of the non-conservative (auxiliary) variables using the
        :class:`~.EOS`
        """

        # Relaxation to update the volume fraction
        self.relaxation(values)

        # auxilliary variables update
        self.auxilliaryVariableUpdate(values)

    def CFL(
        self,
        cells: MeshCellSet,
        CFL_value,
    ) -> float:
        dt = super().CFL(cells, CFL_value)

        dx = cells.min_length

        # Get the velocity components
        UV_slice = slice(Q.fields.U, Q.fields.V + 1)
        UV = cells.values[..., [0], UV_slice]

        U = np.linalg.norm(UV, axis=-1)

        max_vel = np.max(
            self.compute_max_vel(U, cells.values.view(Q), self.problem.sigma)
        )

        dt = np.min((dt, CFL_value * dx / max_vel))

        return dt

    def compute_max_vel(self, U: np.ndarray, values: Q, sigma: float):
        cFd = values[..., [0], Q.fields.cFd]
        norm_grada = values[..., [0], Q.fields.norm_grada]
        rho = values[..., [0], Q.fields.rho]
        if np.any(cFd<=0):
            print(cFd[np.where(cFd<=0)])
            exit()
        if np.any(rho<=0):
            print(rho[np.where(rho<=0)])
            exit()
        if np.any(norm_grada<0):
            exit()
        if np.any(U<0):
            exit()
        # r = Ma ** 2 / We
        # c_cap = cFd * (1 + r / 8)
        return U + cFd * (1 + (sigma * norm_grada) / (cFd**2 * rho) / 8)


    # def prim2Qc(self, values: Q):
    #    values[..., Q.fields.abarrho] = values[..., Q.fields.abar] * (
    #        values[..., Q.fields.arho1]
    #        + values[..., Q.fields.arho2]
    #        + values[..., Q.fields.arho1d]
    #    )

    def prim2Qc(self, values: Q):
        abar = values[..., Q.fields.abar]
        U = values[..., Q.fields.U]
        V = values[..., Q.fields.V]
        pbar = values[..., Q.fields.pbar]
        arho1d = values[..., Q.fields.arho1d]
        ad = values[..., Q.fields.ad]
        H = values[..., Q.fields.H]
        sigma = self.problem.sigma

        # For linearized EOS only
        p1 = np.full_like(abar, np.nan)
        p1 = np.where(
            abar > 0,
            np.where(~np.isnan(H), pbar + (1 - abar) * sigma * H,pbar),
            # pbar + (1 - abar) * sigma * H,
            np.nan,
        )
        rho1 = self.problem.eos[Phases.PHASE1].rho(p1)
        p2 = np.full_like(abar, np.nan)
        p2 = np.where(
            abar < 1,
            # pbar - abar * sigma * H,
            np.where(~np.isnan(H), pbar - abar * sigma * H,pbar),
            np.nan,
        )
        rho2 = self.problem.eos[Phases.PHASE2].rho(p2)

        values[..., Q.fields.arho1] = np.where(
            abar > 0,
            abar * (1 - ad) * rho1,
            0,
        )
        values[..., Q.fields.arho2] = np.where(
            (1 - abar) > 0,
            (1 - abar) * (1 - ad) * rho2,
            0,
        )
        rho = values[..., Q.fields.arho1] + values[..., Q.fields.arho2] + arho1d

        values[..., Q.fields.abarrho] = abar * rho
        values[..., Q.fields.rhoU] = rho * U
        values[..., Q.fields.rhoV] = rho * V

    def post_extrapolation(self, values: Q):
        self.prim2Qc(values)

        self.geoUpdate = False
        self.relaxation(values)
        self.geoUpdate = True

        # auxilliary variables update
        self.auxilliaryVariableUpdateNoGeo(values)
        # self.auxilliaryVariableUpdate(values)


class Rusanov(TsCapScheme):
    def intercellFlux(
        self,
        Q_L: Q,
        Q_R: Q,
        normals: np.ndarray,
        surfaces: np.ndarray,
    ):
        r"""This schemes implements the Rusanov scheme for a
        :class:`TwoPhaseProblem`. It applies the :class:`~.euler.Rusanov`
        scheme indipendently for each phase (with the :math:`\sigma` correctly
        calculated among all the two phases state)

        Parameters
        ----------
        Q_L:
            State values in the "left" side of the interface

        Q_R:
            State values in the "right" side of the interface

        Returns
        -------
        F
            The value of the numerical convective flux multiplied by the
            surface value :math:`\numConvective`
        """

        FS = np.zeros_like(Q_L).view(Q)
        fields = Q.fields

        # Get normal velocities
        Un_L = np.einsum(
            "...l,...->...l", Q_L[..., fields.U], normals[..., 0]
        ) + np.einsum("...l,...->...l", Q_L[..., fields.V], normals[..., 1])
        Un_R = np.einsum(
            "...l,...->...l", Q_R[..., fields.U], normals[..., 0]
        ) + np.einsum("...l,...->...l", Q_R[..., fields.V], normals[..., 1])
        max_vel = np.maximum(
            self.compute_max_vel(Un_L, Q_L, self.problem.sigma),
            self.compute_max_vel(Un_R, Q_R, self.problem.sigma),
        )

        DeltaF = 0.5 * (self.problem.F(Q_L) + self.problem.F(Q_R))

        # This is the flux tensor dot the normal
        DeltaF = np.einsum("...mkl,...l->...mk", DeltaF, normals)

        Qc_L = Q_L.view(Q).get_conservative()
        Qc_R = Q_R.view(Q).get_conservative()

        DeltaQ = np.einsum(  # type: ignore
            "...k,...kl->...kl", 0.5 * max_vel, (Qc_R - Qc_L)
        )

        FS.view(Q).set_conservative(
            surfaces[..., np.newaxis, np.newaxis] * (DeltaF - DeltaQ)
        )

        return FS

    def post_extrapolation(self, values: Q):
        self.prim2Qc(values)

        self.geoUpdate = False
        self.relaxation(values)
        self.geoUpdate = True

        # auxilliary variables update
        self.auxilliaryVariableUpdateNoGeo(values)
        # self.auxilliaryVariableUpdate(values)
