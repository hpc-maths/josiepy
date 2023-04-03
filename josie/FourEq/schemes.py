# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from josie.mesh.cellset import MeshCellSet
from josie.euler.schemes import Rusanov as EulerRusanov
from josie.scheme import Scheme
from josie.scheme.convective import ConvectiveScheme
from josie.twofluid.state import PhasePair
from josie.twofluid.fields import Phases

from .eos import TwoPhaseEOS
from .problem import FourEqProblem
from .state import Q, FourEqPhaseFields


class FourEqScheme(Scheme):
    """A base class for the four equations twophase scheme"""

    problem: FourEqProblem

    def __init__(self, eos: TwoPhaseEOS, do_relaxation: bool):
        super().__init__(FourEqProblem(eos))
        self.do_relaxation = do_relaxation

    """Ad hoc relaxation for linearized EOS"""

    def relaxForLinearizedEOS(self, values: Q):
        fields = Q.fields

        arho1 = values[..., fields.arho1]
        arho2 = values[..., fields.arho2]

        rho10 = self.problem.eos[Phases.PHASE1].rho0
        rho20 = self.problem.eos[Phases.PHASE2].rho0
        c1 = self.problem.eos[Phases.PHASE1].c0
        c2 = self.problem.eos[Phases.PHASE2].c0

        q = rho20 * c2**2 - rho10 * c1**2

        qtilde = arho2 * c2**2 - arho1 * c1**2

        betaPos = (
            q
            - qtilde
            + np.sqrt(
                (q - qtilde) ** 2 + 4.0 * arho1 * c1**2 * arho2 * c2**2
            )
        ) / (2.0 * arho2 * c2**2)

        alpha = betaPos / (1.0 + betaPos)
        values[..., fields.alpha] = alpha
        values[..., fields.arho] = alpha * (arho1 + arho2)

    """General relaxation procedure for all other EOS"""

    def relaxation(self, values: Q):
        fields = Q.fields

        alpha = values[..., fields.alpha]
        arho1 = values[..., fields.arho1]
        arho2 = values[..., fields.arho2]

        # Solve for alpha using p1(arho1/alpha) = p2(arho2/alpha) with Newton
        # Note that arho1 and arho2 remain constant
        def phi(arho1: np.ndarray, arho2: np.ndarray, alpha: np.ndarray):
            return self.problem.eos[Phases.PHASE1].p(
                arho1 / alpha
            ) - self.problem.eos[Phases.PHASE2].p(arho2 / (1.0 - alpha))

        def dphi_dalpha(
            arho1: np.ndarray, arho2: np.ndarray, alpha: np.ndarray
        ):
            # Note that dp_drho = c^2 for barotropic EOS
            return (
                -arho1
                / (alpha**2)
                * self.problem.eos[Phases.PHASE1].sound_velocity(arho1 / alpha)
                ** 2
                - arho2
                / ((1.0 - alpha) ** 2)
                * self.problem.eos[Phases.PHASE2].sound_velocity(
                    arho2 / (1.0 - alpha)
                )
                ** 2
            )

        dalpha = 1.0
        iter = 0
        while np.any(dalpha / alpha > 1e-8):
            iter += 1
            dalpha = -phi(arho1, arho2, alpha) / dphi_dalpha(
                arho1, arho2, alpha
            )
            alpha += dalpha
        if np.max(alpha) > 1.0 or np.min(alpha) < 0.0:
            exit()

        values[..., fields.alpha] = alpha
        values[..., fields.arho] = alpha * (arho1 + arho2)

    def auxilliaryVariableUpdate(self, values: Q):
        fields = Q.fields

        arho = values[..., fields.arho]
        arho1 = values[..., fields.arho1]
        arho2 = values[..., fields.arho2]
        rhoU = values[..., fields.rhoU]
        rhoV = values[..., fields.rhoV]

        rho = arho1 + arho2
        alpha = arho / rho

        alphas = PhasePair(alpha, 1.0 - alpha)
        arhos = PhasePair(arho1, arho2)

        values[..., fields.alpha] = alpha
        values[..., fields.rho] = rho
        values[..., fields.U] = rhoU / rho
        values[..., fields.V] = rhoV / rho

        c_sq = 0.0  # Auxliary variable for mixture speed of sound

        for phase in Phases:
            phase_values = values.view(Q).get_phase(phase)

            alpha = alphas[phase]
            arho = arhos[phase]

            rho = arho / alpha
            p = self.problem.eos[phase].p(rho)
            c = self.problem.eos[phase].sound_velocity(rho)

            phase_values[..., FourEqPhaseFields.p] = p
            phase_values[..., FourEqPhaseFields.c] = c

            values.view(Q).set_phase(
                phase,
                phase_values,
            )

            c_sq += arho * c**2

        values[..., fields.c] = np.sqrt(c_sq / values[..., fields.rho])
        values[..., fields.P] = (
            alpha * values[..., fields.p1]
            + (1 - alpha) * values[..., fields.p2]
        )

    def post_extrapolation(self, values: Q):
        # auxilliary variables update
        super().post_extrapolation(values)

        if self.do_relaxation:
            # Relaxation to update the volume fraction
            if np.all(
                [
                    self.problem.eos[phase].__class__.__name__
                    == "LinearizedGas"
                    for phase in Phases
                ]
            ):
                self.relaxForLinearizedEOS(values)
            else:
                self.relaxation(values)

        # auxilliary variables update
        self.auxilliaryVariableUpdate(values)

    def post_step(self, values: Q):
        """During the step we update the conservative values. After the
        step we update the non-conservative variables. This method updates
        the values of the non-conservative (auxiliary) variables using the
        :class:`~.EOS`
        """

        # auxilliary variables update
        self.auxilliaryVariableUpdate(values)

        if self.do_relaxation:
            # Relaxation bto update the volume fraction
            if np.all(
                [
                    self.problem.eos[phase].__class__.__name__
                    == "LinearizedGas"
                    for phase in Phases
                ]
            ):
                self.relaxForLinearizedEOS(values)
            else:
                self.relaxation(values)

        # auxilliary variables update
        self.auxilliaryVariableUpdate(values)


class Rusanov(ConvectiveScheme, FourEqScheme):
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
        c = Q_L[..., fields.c]
        c_neigh = Q_R[..., fields.c]

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

        sigma = np.max(np.abs(U) + c[..., np.newaxis])

        dt = np.min((dt, CFL_value * dx / sigma))

        return dt
