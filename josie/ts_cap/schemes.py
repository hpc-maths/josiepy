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

    def __init__(
        self,
        eos: TwoPhaseEOS,
        sigma: float,
        Hmax: float,
        dx: float,
        dy: float,
        norm_grada_min: float,
        nSmoothPass: int,
    ):
        super().__init__(TsCapProblem(eos, sigma, Hmax, norm_grada_min))

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

            # Recursive call
            return self.remove_outlier(field, n - 1)

    def relaxation(self, values: Q):
        fields = Q.fields

        arho1 = values[..., fields.arho1]
        arho2 = values[..., fields.arho2]
        arho1d = values[..., fields.arho1d]
        rho1d = self.problem.eos[Phases.PHASE1].rho0
        ad = values[..., fields.ad]
        capSigma = values[..., fields.capSigma]
        abarrho = values[..., fields.abarrho]
        rho = arho1 + arho2 + arho1d

        # Compute estimator of the relaxation within [0,1]
        abar = np.minimum(np.maximum(abarrho / rho, 0), 1)

        values[..., fields.abar] = abar
        self.updateGeometry(values)
        H = values[..., fields.H]
        Hlim = np.minimum(H, self.problem.Hmax)
        DH = H - Hlim

        # Note that rho remain constant
        def phi(
            arho1: np.ndarray,
            arho2: np.ndarray,
            abar: np.ndarray,
            ad: np.ndarray,
            Hlim: np.ndarray,
        ):
            rho1 = np.full_like(arho1, np.nan)
            rho2 = np.full_like(arho1, np.nan)
            np.divide(arho1, abar * (1 - ad), where=(abar > 0) & (arho1 > 0), out=rho1)
            np.divide(
                arho2,
                (1 - abar) * (1 - ad),
                where=((1.0 - abar) > 0) & (arho2 > 0),
                out=rho2,
            )
            self.phi_out = np.where(
                (abar > 0) & (arho1 > 0),
                self.problem.eos[Phases.PHASE1].p(rho1),
                np.nan,
            )
            self.phi_out -= np.where(
                ((1.0 - abar) > 0) & (arho2 > 0),
                self.problem.eos[Phases.PHASE2].p(rho2),
                np.nan,
            )
            return (1 - ad) * self.phi_out - self.problem.sigma * Hlim
            # return (1 - ad) * (
            #     self.problem.eos[Phases.PHASE1].p(rho1)
            #     - self.problem.eos[Phases.PHASE2].p(rho2)
            # ) - self.problem.sigma * Hlim

        def dphi_dabar(
            arho1: np.ndarray, arho2: np.ndarray, abar: np.ndarray, ad: np.ndarray
        ):
            # Note that dp_drho = c^2 for barotropic EOS
            return (
                -arho1
                / (abar**2)
                * self.problem.eos[Phases.PHASE1].sound_velocity(
                    arho1 / abar / (1 - ad)
                )
                ** 2
                - arho2
                / ((1.0 - abar) ** 2)
                * self.problem.eos[Phases.PHASE2].sound_velocity(
                    arho2 / (1.0 - abar) / (1 - ad)
                )
                ** 2
            )

        def dphi_dm1(arho1: np.ndarray, abar: np.ndarray, ad: np.ndarray):
            rho1 = np.full_like(arho1, np.nan)
            np.divide(arho1, abar * (1 - ad), where=(abar > 0) & (arho1 > 0), out=rho1)
            return self.problem.eos[Phases.PHASE1].sound_velocity(rho1) ** 2 / abar

        def dphi_dad(
            arho1: np.ndarray,
            arho2: np.ndarray,
            abar: np.ndarray,
            ad: np.ndarray,
        ):
            rho1 = np.full_like(arho1, np.nan)
            rho2 = np.full_like(arho1, np.nan)
            np.divide(arho1, abar * (1 - ad), where=(abar > 0) & (arho1 > 0), out=rho1)
            np.divide(
                arho2,
                (1 - abar) * (1 - ad),
                where=((1 - abar) > 0) & (arho2 > 0),
                out=rho2,
            )
            p1 = self.problem.eos[Phases.PHASE1].p(rho1)
            c1 = self.problem.eos[Phases.PHASE1].sound_velocity(rho1)
            p2 = self.problem.eos[Phases.PHASE2].p(rho2)
            c2 = self.problem.eos[Phases.PHASE2].sound_velocity(rho2)
            return p2 - p1 + c1**2 * rho1 - c2**2 * rho2

        # Init NR method
        dabar = np.zeros_like(abar)
        iter = 0

        # Index that locates the cell where there the pressures need to be relaxed
        eps = 1e-9
        tol = 1e-5
        p0 = self.problem.eos[Phases.PHASE1].p0
        index = np.where(
            (np.abs(phi(arho1, arho2, abar, ad, Hlim)) > tol * p0)
            & (abar > eps)
            & (1 - abar > eps)
        )
        while index[0].size > 0:
            # Counter
            iter += 1

            # Mass transfer modification of the jacobian
            R = (
                arho1[index]
                / abar[index]
                / (1 - abar[index])
                * self.problem.sigma
                * DH[index]
            )
            F = phi(arho1[index], arho2[index], abar[index], ad[index], Hlim[index])
            dphi_trans = (
                R
                / F
                * (
                    dphi_dad(arho1[index], arho2[index], abar[index], ad[index]) / rho1d
                    - dphi_dm1(arho1[index], abar[index], ad[index])
                )
            )
            # NR step
            dabar[index] = -F / (
                dphi_dabar(arho1[index], arho2[index], abar[index], ad[index])
                + dphi_trans
            )

            # Prevent the NR method to explore out of the interval [0,1]
            dabar[index] = np.where(
                dabar[index] < 0,
                np.maximum(dabar[index], -0.9 * abar[index]),
                np.minimum(dabar[index], 0.9 * (1 - abar[index])),
            )

            # Update values
            abar[index] += dabar[index]
            arho1[index] += -dabar[index] / F * R
            # capSigma += dabar / f * R * capSigma[index] / md[index]
            # il faut fixer une taille moyenne de depart
            arho1d[index] += dabar[index] / F * R
            ad[index] += dabar[index] / F * R / rho1d

            # Update stored values
            values[..., fields.arho1] = arho1
            values[..., fields.arho1d] = arho1d
            values[..., fields.abarrho] = abar * rho
            values[..., fields.ad] = ad
            values[..., fields.capSigma] = capSigma

            # Update the index where the NR method is applied
            index = np.where(
                (np.abs(phi(arho1, arho2, abar, ad, Hlim)) > tol * p0)
                & (abar > eps)
                & (1 - abar > eps)
            )

            # Safety check
            if iter > 20:
                exit()

    def updateGeometry(self, values: Q):
        fields = Q.fields
        dx = self.dx
        dy = self.dy

        # Geometric periodic update
        abar = values[..., fields.abar]
        # Left
        abar[0, ...] = abar[-2, ...]
        # Right
        abar[-1, ...] = abar[1, ...]
        # Top
        abar[:, 0, ...] = abar[:, -2, ...]
        # Bottom
        abar[:, -1, ...] = abar[:, 1, ...]
        grada_x, grada_y = np.gradient(abar, dx, dy, axis=(0, 1))

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
        norm_grada = np.sqrt(grada_x**2 + grada_y**2)
        n_x = np.full_like(grada_x, np.nan)
        n_y = np.full_like(grada_y, np.nan)
        np.divide(
            grada_x, norm_grada, where=norm_grada > self.problem.norm_grada_min, out=n_x
        )
        np.divide(
            grada_y, norm_grada, where=norm_grada > self.problem.norm_grada_min, out=n_y
        )
        ind = np.where((norm_grada > 0) & (np.isnan(n_x)))
        if ind[0].size > 0:
            exit()

        H = -(self.nan_gradient(n_x, dx, dy)[0] + self.nan_gradient(n_y, dx, dy)[1])

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
        sigma = self.problem.sigma

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
        cFd = np.sqrt(c_sq / rho) / (1 - ad)

        MaX = U / cFd
        MaY = V / cFd
        norm_grada = values[..., fields.norm_grada]
        n_x = values[..., fields.n_x]
        n_y = values[..., fields.n_y]
        WeX = np.empty_like(norm_grada)
        WeX[:] = np.nan
        np.divide(sigma * norm_grada, rho * U**2, where=U**2 > 0, out=WeX)
        WeY = np.empty_like(norm_grada)
        WeY[:] = np.nan
        np.divide(sigma * norm_grada, rho * V**2, where=V**2 > 0, out=WeY)
        r = np.empty_like(norm_grada)
        r[:] = np.nan
        np.divide(MaX**2, WeX, where=WeX > 0, out=r)
        c_cap1X = cFd * (1 + 0.5 * r * n_x**2 * (1 - n_x**2))
        c_cap2X = cFd * (1 - n_x**2) * np.sqrt(r)
        r[:] = np.nan
        np.divide(MaY**2, WeY, where=WeY > 0, out=r)
        c_cap1Y = cFd * (1 + 0.5 * r * n_y**2 * (1 - n_y**2))
        c_cap2Y = cFd * (1 - n_y**2) * np.sqrt(r)

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
        values[..., fields.MaX] = MaX
        values[..., fields.MaY] = MaY
        values[..., fields.WeX] = WeX
        values[..., fields.WeY] = WeY
        values[..., fields.c_cap1X] = c_cap1X
        values[..., fields.c_cap1Y] = c_cap1Y
        values[..., fields.c_cap2X] = c_cap2X
        values[..., fields.c_cap2Y] = c_cap2Y

    def auxilliaryVariableUpdateNoGeo(self, values: Q):
        fields = Q.fields
        sigma = self.problem.sigma

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
        cFd = np.sqrt(c_sq / rho) / (1 - ad)

        MaX = U / cFd
        MaY = V / cFd
        norm_grada = values[..., fields.norm_grada]
        n_x = values[..., fields.n_x]
        n_y = values[..., fields.n_y]
        WeX = np.empty_like(norm_grada)
        WeX[:] = np.nan
        np.divide(sigma * norm_grada, rho * U**2, where=U**2 > 0, out=WeX)
        WeY = np.empty_like(norm_grada)
        WeY[:] = np.nan
        np.divide(sigma * norm_grada, rho * V**2, where=V**2 > 0, out=WeY)
        r = np.empty_like(norm_grada)
        r[:] = np.nan
        np.divide(MaX**2, WeX, where=WeX > 0, out=r)
        c_cap1X = cFd * (1 + 0.5 * r * n_x**2 * (1 - n_x**2))
        c_cap2X = cFd * (1 - n_x**2) * np.sqrt(r)
        r[:] = np.nan
        np.divide(MaY**2, WeY, where=WeY > 0, out=r)
        c_cap1Y = cFd * (1 + 0.5 * r * n_y**2 * (1 - n_y**2))
        c_cap2Y = cFd * (1 - n_y**2) * np.sqrt(r)

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
        values[..., fields.MaX] = MaX
        values[..., fields.MaY] = MaY
        values[..., fields.WeX] = WeX
        values[..., fields.WeY] = WeY
        values[..., fields.c_cap1X] = c_cap1X
        values[..., fields.c_cap1Y] = c_cap1Y
        values[..., fields.c_cap2X] = c_cap2X
        values[..., fields.c_cap2Y] = c_cap2Y

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

        Ma = U / cFd

        We = np.full_like(Ma, np.inf)
        np.divide(
            values[..., [0], Q.fields.rho] * U**2,
            sigma * values[..., [0], Q.fields.norm_grada],
            where=sigma * values[..., [0], Q.fields.norm_grada] > 0,
            out=We,
        )
        r = np.zeros_like(Ma)
        r = np.divide(Ma**2, We, where=We > 0, out=r)
        if np.any(
            np.isnan(
                np.maximum(
                    np.abs(U - cFd * (1 + r / 8)),
                    np.abs(U + cFd * (1 + r / 8)),
                )
            )
        ):
            np.set_printoptions(linewidth=250)
            print(U[..., 0])
            exit()

        return np.maximum(
            np.abs(U - cFd * (1 + r / 8)),
            np.abs(U + cFd * (1 + r / 8)),
        )

    def prim2Qc(self, values: Q):
        rho = values[..., Q.fields.rho]
        U = values[..., Q.fields.U]
        V = values[..., Q.fields.V]
        abar = values[..., Q.fields.abar]
        arho1d = values[..., Q.fields.arho1d]
        ad = values[..., Q.fields.ad]
        H = values[..., Q.fields.H]
        sigma = self.problem.sigma

        if np.any(rho <= 0):
            exit()

        # For linearized EOS only
        c1 = self.problem.eos[Phases.PHASE1].c0
        rho01 = self.problem.eos[Phases.PHASE1].rho0
        c2 = self.problem.eos[Phases.PHASE2].c0
        rho02 = self.problem.eos[Phases.PHASE2].rho0
        # rho1, rho2 solution of
        #       rho = abar (1-ad) rho1 + (1-abar) (1-ad) rho2 + arho1d
        # and   p1(rho1)-p2(rho2) = 0

        rho1 = np.full_like(rho, np.nan)
        rho1 = np.where(
            (abar > 0),
            np.where(
                ~np.isnan(H),
                (
                    c2**2 * (rho - arho1d - (1 - abar) * (1 - ad) * rho02)
                    + (1 - abar) * (1 - ad) * c1**2 * rho01
                    + (1 - abar) * (1 - ad) * sigma * H
                )
                / ((1 - ad) * ((1 - abar) * c1**2 + abar * c2**2)),
                (
                    c2**2 * (rho - arho1d - (1 - abar) * (1 - ad) * rho02)
                    + (1 - abar) * (1 - ad) * c1**2 * rho01
                )
                / ((1 - ad) * ((1 - abar) * c1**2 + abar * c2**2)),
            ),
            np.nan,
        )
        rho2 = np.full_like(rho, np.nan)
        rho2 = np.where(
            (1 - abar) > 0,
            np.where(
                ~np.isnan(H),
                (
                    c1**2 * (rho - arho1d - abar * (1 - ad) * rho01)
                    + abar * (1 - ad) * c2**2 * rho02
                    + abar * (1 - ad) * sigma * H
                )
                / ((1 - ad) * (abar * c2**2 + (1 - abar) * c1**2)),
                (
                    c1**2 * (rho - arho1d - abar * (1 - ad) * rho01)
                    + abar * (1 - ad) * c2**2 * rho02
                )
                / ((1 - ad) * (abar * c2**2 + (1 - abar) * c1**2)),
            ),
            np.nan,
        )

        values[..., Q.fields.abarrho] = abar * rho
        values[..., Q.fields.rhoU] = rho * U
        values[..., Q.fields.rhoV] = rho * V
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
        # self.relaxation(values)

        # auxilliary variables update
        self.auxilliaryVariableUpdateNoGeo(values)
        # self.auxilliaryVariableUpdate(values)
