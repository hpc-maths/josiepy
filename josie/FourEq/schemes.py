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
from .problem import FourEqProblem
from .state import Q, FourEqPhaseFields


class FourEqScheme(ConvectiveScheme):
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
            + np.sqrt((q - qtilde) ** 2 + 4.0 * arho1 * c1**2 * arho2 * c2**2)
        ) / (2.0 * arho2 * c2**2)

        alpha = betaPos / (1.0 + betaPos)
        values[..., fields.alpha] = alpha
        values[..., fields.arho] = alpha * (arho1 + arho2)

    """General relaxation procedure for all other EOS"""

    def relaxation(self, values: Q):
        fields = Q.fields

        arho1 = values[..., fields.arho1]
        arho2 = values[..., fields.arho2]

        # Compute estimator of the relaxation within [0,1]
        alpha = np.minimum(np.maximum(values[..., fields.arho] / (arho1 + arho2), 0), 1)

        # Solve for alpha using p1(arho1/alpha) = p2(arho2/alpha) with Newton
        # Note that arho1 and arho2 remain constant
        def phi(arho1: np.ndarray, arho2: np.ndarray, alpha: np.ndarray):
            rho1 = np.full_like(arho1, np.nan)
            rho2 = np.full_like(arho1, np.nan)
            np.divide(arho1, alpha, where=(alpha > 0) & (arho1 > 0), out=rho1)
            np.divide(
                arho2, 1.0 - alpha, where=(1.0 - alpha > 0) & (arho2 > 0), out=rho2
            )
            return self.problem.eos[Phases.PHASE1].p(rho1) - self.problem.eos[
                Phases.PHASE2
            ].p(rho2)

        def dphi_dalpha(arho1: np.ndarray, arho2: np.ndarray, alpha: np.ndarray):
            # Note that dp_drho = c^2 for barotropic EOS
            return (
                -arho1
                / (alpha**2)
                * self.problem.eos[Phases.PHASE1].sound_velocity(arho1 / alpha) ** 2
                - arho2
                / ((1.0 - alpha) ** 2)
                * self.problem.eos[Phases.PHASE2].sound_velocity(arho2 / (1.0 - alpha))
                ** 2
            )

        # Init NR method
        dalpha = np.zeros_like(alpha)
        iter = 0

        # Index that locates the cell where there the pressures need to be relaxed
        eps = 1e-9
        index = np.where(np.abs(phi(arho1, arho2, alpha)) > eps * 1e5)
        while index[0].size > 0:
            # Counter
            iter += 1

            # NR step
            dalpha[index] = -phi(
                arho1[index], arho2[index], alpha[index]
            ) / dphi_dalpha(arho1[index], arho2[index], alpha[index])

            # Prevent the NR method to explore out of the interval [0,1]
            alpha[index] += np.where(
                dalpha[index] < 0,
                np.maximum(dalpha[index], -0.9 * alpha[index]),
                np.minimum(dalpha[index], 0.9 * (1 - alpha[index])),
            )
            tol = 1e-6
            alpha = np.where(alpha < tol, 0, alpha)
            alpha = np.where(1 - alpha < tol, 1, alpha)

            # Update the index where the NR method is applied
            index = np.where((np.abs(phi(arho1, arho2, alpha)) > eps * 1e5))

            # Safety check
            if iter > 50:
                exit()

        # Update the alpha-dependent conservative field
        values[..., fields.arho] = alpha * (arho1 + arho2)

    def prim2Q(self, values: Q):
        fields = Q.fields

        rho = values[..., fields.rho]
        P = values[..., fields.P]
        U = values[..., fields.U]
        V = values[..., fields.V]

        rho1 = self.problem.eos[Phases.PHASE1].rho(P)
        rho2 = self.problem.eos[Phases.PHASE2].rho(P)
        c1 = self.problem.eos[Phases.PHASE1].sound_velocity(rho1)
        c2 = self.problem.eos[Phases.PHASE2].sound_velocity(rho2)

        alpha = np.minimum(np.maximum((rho - rho2) / (rho1 - rho2), 0), 1)
        # alpha = (rho - rho2) / (rho1 - rho2)

        if np.any(alpha < 0.0) or np.any(alpha > 1.0) or np.any(np.isnan(alpha)):
            print(alpha)
            exit()
        if np.any(rho < 0):
            print(rho)
            exit()

        values[..., fields.arho] = alpha * rho
        values[..., fields.arho1] = alpha * rho1
        values[..., fields.arho2] = (1 - alpha) * rho2
        values[..., fields.rhoU] = rho * U
        values[..., fields.rhoV] = rho * V

        values[..., fields.c] = np.sqrt(
            (alpha * rho1 * c1**2 + (1 - alpha) * rho2 * c2**2) / rho
        )
        values[..., fields.alpha] = alpha
        values[..., fields.p1] = P
        values[..., fields.p2] = P
        values[..., fields.c1] = c1
        values[..., fields.c2] = c2

    def auxilliaryVariableUpdate(self, values: Q):
        fields = Q.fields

        arho = values[..., fields.arho]
        arho1 = values[..., fields.arho1]
        arho2 = values[..., fields.arho2]
        rhoU = values[..., fields.rhoU]
        rhoV = values[..., fields.rhoV]

        rho = arho1 + arho2
        alpha = arho / rho

        if np.any(alpha < 0.0):
            print(alpha)
            exit()
        if np.any(alpha > 1.0):
            print(alpha)
            exit()
        if np.any(np.isnan(alpha)):
            print(alpha)
            exit()

        alphas = PhasePair(alpha, 1.0 - alpha)
        arhos = PhasePair(arho1, arho2)

        values[..., fields.alpha] = alpha
        values[..., fields.rho] = rho
        values[..., fields.U] = rhoU / rho
        values[..., fields.V] = rhoV / rho

        c_sq = 0.0  # Auxiliary variable for mixture speed of sound

        for phase in Phases:
            phase_values = values.view(Q).get_phase(phase)

            alpha = alphas[phase]
            arho = arhos[phase]

            rho = np.full_like(alpha, np.nan)
            np.divide(arho, alpha, where=alpha > 0, out=rho)
            p = self.problem.eos[phase].p(rho)
            c = self.problem.eos[phase].sound_velocity(rho)

            phase_values[..., FourEqPhaseFields.p] = p
            phase_values[..., FourEqPhaseFields.c] = c

            values.view(Q).set_phase(
                phase,
                phase_values,
            )
            if np.invert(np.isnan(c)):
                c_sq += arho * c**2

        values[..., fields.c] = np.sqrt(c_sq / values[..., fields.rho])

        alpha = values[..., fields.alpha]
        p1 = values[..., fields.p1]
        p2 = values[..., fields.p2]
        values[..., fields.P] = np.nan
        values[..., fields.P] = np.where(
            (alpha > 0) * (alpha < 1),
            alpha * p1 + (1 - alpha) * p2,
            np.where(alpha == 0, p2, p1),
        )

    def post_extrapolation(self, values: Q):
        self.prim2Q(values)
        # self.relaxation(values)

        # auxilliary variables update
        # self.auxilliaryVariableUpdate(values)

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


class Rusanov(FourEqScheme):
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
        UV = cells.values[..., [0], UV_slice]

        U = np.linalg.norm(UV, axis=-1)
        c = cells.values[..., [0], Q.fields.c]

        sigma = np.max(np.abs(U) + c[..., np.newaxis])

        dt = np.min((dt, CFL_value * dx / sigma))

        return dt
