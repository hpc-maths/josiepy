# josiepy
# Copyright © 2021 Ruben Di Battista
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY Ruben Di Battista ''AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL Ruben Di Battista BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# The views and conclusions contained in the software and documentation
# are those of the authors and should not be interpreted as representing
# official policies, either expressed or implied, of Ruben Di Battista.
""" :class:`BoundaryCondition` classes that are specific to the Navier-Stokes
system """
from __future__ import annotations

from typing import TYPE_CHECKING
from josie.bc import BoundaryCondition, Dirichlet, Neumann
from josie.euler.eos import EOS
from josie.euler.fields import EulerFields as fields


if TYPE_CHECKING:
    from josie.bc import ImposedValue
    from josie.mesh.cellset import MeshCellSet
    from josie.boundary import Boundary


class Inlet(BoundaryCondition):
    """Imposes an inlet boundary condition. I.e. velocity and internal energy
    are imposed, for pressure a zero gradient condition is set.

    Parameters
    ----------
    U
        x-component of the flow velocity to impose, as a :class:`Callable`
        taking  :class:`CellSet` instance as parameter referring to the cells
        of the related boundary
    V
        y-component of the flow velocity to impose, as a :class:`Callable`
        taking  :class:`CellSet` as parameter referring to the cells of the
        related boundary
    e
        The internal energy to impose, as a :class:`Callable` taking
        :class:`CellSet` as parameter referring to the cells of the related
        boundary
    eos
        The same :class:`EOS` used for the rest of the problem

    constant
        Set this flag to ``True`` to explicitly force the creation of a
        constant boundary condition. A constant BC is optimized to reduce the
        number of calls.

        :class:`Dirichlet` normally is capable to understand automatically if
        you are providing a constant imposed value: if you provide a constant
        scalar :class:`float` or :class:`int` (or a :class:`State` containing a
        scalar value for each field). If you provide a :class:`BCCallable` then
        it cannot automatically infer it your callable is actually only a
        function of space (i.e. it does not change at every time step) or not.
        If you want to optimize the call, you need to explicitly set
        ``constant`` to ``True``.

    """

    # TODO: Add 3D
    def __init__(
        self,
        U: ImposedValue,
        V: ImposedValue,
        e: ImposedValue,
        eos: EOS,
        constant=True,
    ):
        # The partial set of BCs to impose
        self.U_bc = Dirichlet(U, constant)
        self.V_bc = Dirichlet(V, constant)
        self.e_bc = Dirichlet(e, constant)

        self.zero_gradient = Neumann(0, constant)

        self.eos = eos

    def init(self, cells: MeshCellSet, boundary: Boundary):
        boundary_idx = boundary.cells_idx
        boundary_cells = cells[boundary_idx]

        self.U_bc.init(boundary_cells)
        self.V_bc.init(boundary_cells)
        self.e_bc.init(boundary_cells)

        self.zero_gradient.init(boundary_cells)

    def __call__(self, cells: MeshCellSet, boundary: Boundary, t: float):
        ghost_idx = boundary.ghost_cells_idx
        boundary_idx = boundary.cells_idx

        boundary_cells = cells[boundary_idx]
        ghost_cells = cells[ghost_idx]

        # Let's compute the ghost value of pressure that imposes the zero
        # gradient condition
        p_ghost = self.zero_gradient(boundary_cells, ghost_cells, fields.p, t)

        # And Dirichlet ghost values for internal energy
        e_ghost = self.e_bc(boundary_cells, ghost_cells, fields.e, t)

        # Compute the corresponding density using eos
        rho_ghost = self.eos.rho(p_ghost, e_ghost)

        # And speed of sound
        c_ghost = self.eos.sound_velocity(rho_ghost, p_ghost)

        # Compute all the derived ghost values
        U_ghost = self.U_bc(boundary_cells, ghost_cells, fields.U, t)
        V_ghost = self.V_bc(boundary_cells, ghost_cells, fields.V, t)

        rhoe_ghost = rho_ghost * e_ghost
        rhoE_ghost = rho_ghost * (e_ghost + (U_ghost ** 2 + V_ghost ** 2) / 2)
        rhoU_ghost = rho_ghost * U_ghost
        rhoV_ghost = rho_ghost * V_ghost

        # Impose the ghost values
        for field, ghost_value in (
            (fields.rho, rho_ghost),
            (fields.rhoU, rhoU_ghost),
            (fields.rhoV, rhoV_ghost),
            (fields.rhoE, rhoE_ghost),
            (fields.rhoe, rhoe_ghost),
            (fields.U, U_ghost),
            (fields.V, V_ghost),
            (fields.p, p_ghost),
            (fields.c, c_ghost),
        ):

            cells._values[ghost_idx[0], ghost_idx[1], :, field] = ghost_value


class Outflow(BoundaryCondition):
    """Imposes an outflow boundary condition. I.e. the pressure is imposed, all
    the rest is zero gradient

    Parameters
    ----------
    p
        The pressure to impose

    eos
        The same :class:`EOS` used for the rest of the problem

    """

    def __init__(self, p: float, eos: EOS):
        # The partial set of BCs to impose
        self.p_bc = Dirichlet(p)
        self.zero_gradient = Neumann(0)

        self.eos = eos

    def init(self, cells: MeshCellSet, boundary: Boundary):
        boundary_idx = boundary.cells_idx
        boundary_cells = cells[boundary_idx]

        self.p_bc.init(boundary_cells)
        self.zero_gradient.init(boundary_cells)

    def __call__(self, cells: MeshCellSet, boundary: Boundary, t: float):
        ghost_idx = boundary.ghost_cells_idx
        boundary_idx = boundary.cells_idx

        boundary_cells = cells[boundary_idx]
        ghost_cells = cells[ghost_idx]

        # Let's compute the ghost value of pressure
        p_ghost = self.p_bc(boundary_cells, ghost_cells, fields.p, t)

        # And zero gradient ghost values for internal energy
        e_ghost = self.zero_gradient(boundary_cells, ghost_cells, fields.e, t)

        # Compute the corresponding density using eos
        rho_ghost = self.eos.rho(p_ghost, e_ghost)

        # And speed of sound
        c_ghost = self.eos.sound_velocity(rho_ghost, p_ghost)

        # Compute all the derived ghost values
        U_ghost = self.zero_gradient(boundary_cells, ghost_cells, fields.U, t)
        V_ghost = self.zero_gradient(boundary_cells, ghost_cells, fields.V, t)
        rhoe_ghost = rho_ghost * e_ghost
        rhoE_ghost = rho_ghost * (e_ghost + (U_ghost ** 2 + V_ghost ** 2) / 2)
        rhoU_ghost = rho_ghost * U_ghost
        rhoV_ghost = rho_ghost * V_ghost

        # Impose the ghost values
        for field, ghost_value in (
            (fields.rho, rho_ghost),
            (fields.rhoU, rhoU_ghost),
            (fields.rhoV, rhoV_ghost),
            (fields.rhoE, rhoE_ghost),
            (fields.rhoe, rhoe_ghost),
            (fields.U, U_ghost),
            (fields.V, V_ghost),
            (fields.p, p_ghost),
            (fields.c, c_ghost),
        ):

            cells._values[ghost_idx[0], ghost_idx[1], :, field] = ghost_value


class NoSlip(BoundaryCondition):
    """Imposes no-slip boundary condition. I.e. velocity is zero and the rest
    of variables is zero gradient
    """

    def __init__(self, eos: EOS):
        # The partial set of BCs to impose
        self.no_slip = Dirichlet(0)
        self.zero_gradient = Neumann(0)

        self.eos = eos

    def init(self, cells: MeshCellSet, boundary: Boundary):
        boundary_idx = boundary.cells_idx
        boundary_cells = cells[boundary_idx]

        self.no_slip.init(boundary_cells)
        self.zero_gradient.init(boundary_cells)

    def __call__(self, cells: MeshCellSet, boundary: Boundary, t: float):
        ghost_idx = boundary.ghost_cells_idx
        boundary_idx = boundary.cells_idx

        boundary_cells = cells[boundary_idx]
        ghost_cells = cells[ghost_idx]

        # Let's compute the ghost value of pressure
        p_ghost = self.zero_gradient(boundary_cells, ghost_cells, fields.p, t)

        # And zero gradient ghost values for internal energy
        e_ghost = self.zero_gradient(boundary_cells, ghost_cells, fields.e, t)

        # Compute the corresponding density using eos
        rho_ghost = self.eos.rho(p_ghost, e_ghost)

        # And speed of sound
        c_ghost = self.eos.sound_velocity(rho_ghost, p_ghost)

        # Compute all the derived ghost values
        U_ghost = self.no_slip(boundary_cells, ghost_cells, fields.U, t)
        V_ghost = self.no_slip(boundary_cells, ghost_cells, fields.V, t)
        rhoe_ghost = rho_ghost * e_ghost
        rhoE_ghost = rho_ghost * (e_ghost + (U_ghost ** 2 + V_ghost ** 2) / 2)
        rhoU_ghost = rho_ghost * U_ghost
        rhoV_ghost = rho_ghost * V_ghost

        # Impose the ghost values
        for field, ghost_value in (
            (fields.rho, rho_ghost),
            (fields.rhoU, rhoU_ghost),
            (fields.rhoV, rhoV_ghost),
            (fields.rhoE, rhoE_ghost),
            (fields.rhoe, rhoe_ghost),
            (fields.U, U_ghost),
            (fields.V, V_ghost),
            (fields.p, p_ghost),
            (fields.c, c_ghost),
        ):

            cells._values[ghost_idx[0], ghost_idx[1], :, field] = ghost_value
