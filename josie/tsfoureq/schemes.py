# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from josie.mesh.cellset import MeshCellSet
from josie.euler.schemes import Rusanov as EulerRusanov
from josie.scheme.convective import ConvectiveScheme
from josie.twofluid.state import PhasePair
from josie.twofluid.fields import Phases

from .eos import TwoPhaseEOS
from .problem import TSFourEqProblem
from .state import Q, TSFourEqPhaseFields


class TSFourEqScheme(ConvectiveScheme):
    """A base class for the four equations twophase scheme"""

    problem: TSFourEqProblem

    def __init__(self, eos: TwoPhaseEOS, do_relaxation: bool):
        super().__init__(TSFourEqProblem(eos))
        self.do_relaxation = do_relaxation

    """Ad hoc relaxation for linearized EOS"""

    def relaxForLinearizedEOS(self, values: Q):
        fields = Q.fields

        ad = values[..., fields.ad]
        arho1 = values[..., fields.arho1] / (1 - ad)
        arho2 = values[..., fields.arho2] / (1 - ad)
        arho1d = values[..., fields.arho1d]

        rho10 = self.problem.eos[Phases.PHASE1].rho0
        rho20 = self.problem.eos[Phases.PHASE2].rho0
        c1 = self.problem.eos[Phases.PHASE1].c0
        c2 = self.problem.eos[Phases.PHASE2].c0

        q = rho20 * c2**2 - rho10 * c1**2

        qtilde = arho2 * c2**2 - arho1 * c1**2

        betaPos = (
            q
            - qtilde
            + np.sqrt((q - qtilde) ** 2 + 4.0 * arho1 * c1**2 * arho2 * c2**2)
        ) / (2.0 * arho2 * c2**2)

        alpha = betaPos / (1.0 + betaPos)
        values[..., fields.abar] = alpha
        values[..., fields.abarrho] = alpha * (
            arho1 * (1 - ad) + arho2 * (1 - ad) + arho1d
        )

    """General relaxation procedure for all other EOS"""

    def relaxation(self, values: Q):
        fields = Q.fields

        arho1 = values[..., fields.arho1]
        arho2 = values[..., fields.arho2]
        arho1d = values[..., fields.arho1d]
        ad = values[..., fields.ad]
        abarrho = values[..., fields.abarrho]
        rho = arho1 + arho2 + arho1d

        # Compute estimator of the relaxation within [0,1]
        abar = np.minimum(np.maximum(abarrho / rho, 0), 1)

        values[..., fields.abar] = abar

        # Note that rho remain constant
        def phi(
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
                where=((1.0 - abar) > 0) & (arho2 > 0),
                out=rho2,
            )
            return (1 - ad) * (
                self.problem.eos[Phases.PHASE1].p(rho1)
                - self.problem.eos[Phases.PHASE2].p(rho2)
            )

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

        # Init NR method
        dabar = np.zeros_like(abar)
        iter = 0

        # Index that locates the cell where there the pressures need to be relaxed
        eps = 1e-9
        tol = 1e-5
        p0 = self.problem.eos[Phases.PHASE1].p0
        index = np.where(
            (np.abs(phi(arho1, arho2, abar, ad)) > tol * p0)
            & (abar > eps)
            & (1 - abar > eps)
        )
        while index[0].size > 0:
            # Counter
            iter += 1

            # NR step
            dabar[index] = -phi(arho1[index], arho2[index], abar[index], ad[index]) / (
                dphi_dabar(arho1[index], arho2[index], abar[index], ad[index])
            )

            # Prevent the NR method to explore out of the interval [0,1]
            abar[index] += np.where(
                dabar[index] < 0,
                np.maximum(dabar[index], -0.9 * abar[index]),
                np.minimum(dabar[index], 0.9 * (1 - abar[index])),
            )

            # Update the index where the NR method is applied
            index = np.where(
                (np.abs(phi(arho1, arho2, abar, ad)) > tol * p0)
                & (abar > eps)
                & (1 - abar > eps)
            )

            # Safety check
            if iter > 50:
                exit()

        # Update the abar-dependent conservative field
        values[..., fields.abarrho] = abar * rho

    def auxilliaryVariableUpdate(self, values: Q):
        fields = Q.fields

        abarrho = values[..., fields.abarrho]
        arho1 = values[..., fields.arho1]
        arho2 = values[..., fields.arho2]
        arho1d = values[..., fields.arho1d]
        rhoU = values[..., fields.rhoU]
        rhoV = values[..., fields.rhoV]
        ad = values[..., fields.ad]

        rho = arho1 + arho2 + arho1d
        abar = abarrho / rho

        alphabars = PhasePair(abar, 1.0 - abar)
        arhos = PhasePair(arho1, arho2)

        values[..., fields.abar] = abar
        values[..., fields.rho] = rho
        values[..., fields.U] = rhoU / rho
        values[..., fields.V] = rhoV / rho

        c_sq = 0.0  # Auxiliary variable for mixture speed of sound

        for phase in Phases:
            phase_values = values.view(Q).get_phase(phase)

            alphabar = alphabars[phase]
            arho = arhos[phase]

            rho = np.full_like(arho, np.nan)
            rho = np.divide(arho, alphabar * (1 - ad), where=(alphabar * (1 - ad) > 0))
            p = self.problem.eos[phase].p(rho)
            c = self.problem.eos[phase].sound_velocity(rho)

            phase_values[..., TSFourEqPhaseFields.p] = p
            phase_values[..., TSFourEqPhaseFields.c] = c

            values.view(Q).set_phase(
                phase,
                phase_values,
            )

            c_sq += arho * c**2

        values[..., fields.cFd] = np.sqrt(c_sq / values[..., fields.rho]) / (1 - ad)

        abar = values[..., fields.abar]
        p1 = values[..., fields.p1]
        p2 = values[..., fields.p2]
        values[..., fields.pbar] = np.nan
        values[..., fields.pbar] = np.where(
            (abar > 0) * (abar < 1),
            abar * p1 + (1 - abar) * p2,
            np.where(abar == 0, p2, p1),
        )

    def post_extrapolation(self, values: Q):
        """During the step we update the conservative values. After the
        step we update the non-conservative variables. This method updates
        the values of the non-conservative (auxiliary) variables using the
        :class:`~.EOS`
        """

        # auxilliary variables update
        self.auxilliaryVariableUpdate(values)

        self.relaxation(values)

        # self.prim2Qc(values)

        # auxilliary variables update
        self.auxilliaryVariableUpdate(values)
        # self.relaxation(values)
        # self.auxilliaryVariableUpdate(values)

    def prim2Qc(self, values: Q):
        fields = Q.fields

        rho = values[..., fields.rho]
        pbar = values[..., fields.pbar]
        U = values[..., fields.U]
        V = values[..., fields.V]
        arho1d = values[..., fields.arho1d]
        ad = values[..., fields.ad]

        rho1 = self.problem.eos[Phases.PHASE1].rho(pbar)
        rho2 = self.problem.eos[Phases.PHASE2].rho(pbar)
        abar = (rho - arho1d - (1 - ad) * rho2) / (1 - ad) / (rho1 - rho2)
        abar = np.minimum(np.maximum(abar, 0), 1)

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
        UV = cells.values[..., UV_slice]

        U = np.linalg.norm(UV, axis=-1, keepdims=True)
        c = cells.values[..., Q.fields.cFd]

        sigma = np.max(np.abs(U) + c[..., np.newaxis])

        dt = np.min((dt, CFL_value * dx / sigma))

        return dt


class Rusanov(TSFourEqScheme):
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
        fields = Q.fields

        # Get normal velocities
        UV_slice = slice(fields.U, fields.V + 1)
        UV = Q_L[..., np.newaxis, UV_slice]
        UV_neighs = Q_R[..., np.newaxis, UV_slice]

        # Find the normal velocity
        U = np.einsum("...mkl,...l->...mk", UV, normals)
        U_neigh = np.einsum("...mkl,...l->...mk", UV_neighs, normals)
        c = Q_L[..., fields.cFd]
        c_neigh = Q_R[..., fields.cFd]

        # Let's retrieve the values of the sigma for current cell
        sigma = EulerRusanov.compute_sigma(U, U_neigh, c, c_neigh)

        DeltaF = 0.5 * (self.problem.F(Q_L) + self.problem.F(Q_R))

        # This is the flux tensor dot the normal
        DeltaF = np.einsum("...mkl,...l->...mk", DeltaF, normals)

        values_cons = Q_L.view(Q).get_conservative()
        neigh_values_cons = Q_R.view(Q).get_conservative()

        DeltaQ = 0.5 * sigma * (neigh_values_cons - values_cons)

        FS.view(Q).set_conservative(
            surfaces[..., np.newaxis, np.newaxis] * (DeltaF - DeltaQ)
        )

        return FS
