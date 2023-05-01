# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations


from typing import (
    Callable,
    Sequence,
    List,
    NoReturn,
    Union,
    Type,
    TYPE_CHECKING,
)

from josie.mesh.cellset import CellSet, MeshCellSet

from josie.mesh.cell import DGCell

from josie.state import State
from josie.scheme import Scheme, DGScheme

import numpy as np


if TYPE_CHECKING:
    from josie.mesh.mesh import Mesh


class Solver:
    r"""This class is used to solve a problem governed by PDEs.

    Parameters
    ----------
    mesh
        An instance of the mesh to compute the solution on
    Q
        A :class:`State` representing the variables of the problem
        to be solved

    scheme
        A :class:`Scheme` instance providing the numerical scheme to be
        used for the simulation

    Attributes
    ----------
    t
        The time instant of the simulation held by the :class:`Solver` object
    """

    # Type Checking
    _values: State
    _neighs: Sequence[CellSet]
    t: float

    # TODO: Fix _values to adapt on mesh.dimensionality

    def __init__(self, mesh: Mesh, Q: Type[State], scheme: Scheme):
        self.mesh = mesh
        self.Q = Q
        self.scheme = scheme

        if self.mesh.cell_type == DGCell or issubclass(scheme.__class__, DGScheme):
            raise TypeError("A DGSolver is required for the DG method.")

    def init(self, init_fun: Callable[[MeshCellSet], NoReturn]):
        """
        This method initialize the internal values of the cells of the
        :class:`~.Mesh` and the values of the ghost cells that apply the
        :class:`~.BoundaryCondition` for each boundary of the domain

        Parameters
        ---------
        init_fun
            The function to use to initialize the value in the domain
        """

        # Init time
        self.t = 0.0

        # Init data structure for field values
        self.mesh.cells._values = self.Q.from_mesh(self.mesh)
        self.mesh.cells._values[:] = np.nan

        # First set all the values for the internal cells
        # The actual values are a view of only the internal cells
        init_fun(self.mesh.cells)

        # Note: Corner values are unused

        self.mesh.create_neighbours()
        self.mesh.init_bcs()
        self.mesh.update_ghosts(self.t)

        # Initialize the scheme datastructures (notably the fluxes)
        self.scheme.post_init(self.mesh.cells)

    def copy(self) -> Solver:
        """This methods copies the :class:`Solver` object into another"""

        solver = Solver(self.mesh.copy(), self.Q, self.scheme)
        solver.t = self.t

        return solver

    def step(self, dt: float):
        """This method advances one step in time using the
        :meth:`Scheme.step` method of the given numerical scheme.

        Parameters
        ----------
        dt
            Time increment of the step
        """

        self.scheme.update(self.mesh, dt, self.t)
        self.t += dt

    def plot(self):
        """Plot the current state of the simulation in a GUI."""
        plt = self.mesh.backend
        plt.update(self)

    def animate(self, t):
        """Animate the simulation. Call :meth:`animate` for each time instant
        you want to provide in the animation.

        Parameters
        ----------
        t
            The time instant to animate
        """
        plt = self.mesh.backend
        plt.append(self, t)

    def show(self, fields: Union[List[str], str]):
        """Display on screen the given fields

        Parameters
        ---------
        fields
            The fields you want to plot
        """

        plt = self.mesh.backend
        plt.show(fields)


class DGSolver(Solver):
    dx: float
    dy: float

    def __init__(self, mesh: Mesh, Q: Type[State], scheme: DGScheme):
        self.mesh = mesh
        self.Q = Q
        self.scheme = scheme

        if self.mesh.cell_type != DGCell or not (
            issubclass(scheme.__class__, DGScheme)
        ):
            raise TypeError("A DGCell and a DGScheme are required for the DG method.")

        self.dx = mesh._x[1, 0] - mesh._x[0, 0]
        self.dy = mesh._y[0, 1] - mesh._y[0, 0]

        self.scheme.dx = self.dx
        self.scheme.dy = self.dx

        # Init a local mass matrix in the element of reference
        # Dim : (num_dof, num_dof)
        self.scheme.M_ref = DGCell.refMass()
        # Init a local stiffness matrix in the element of reference
        # Dim : (num_dof, num_dof)
        self.scheme.K_ref = DGCell.refStiff()

        # Init a local edge-mass matrix in the element of reference
        # One matrix for each direction
        # Dim : (num_dof, num_dof)
        self.scheme.eM_ref_tab = DGCell.refMassEdge()

        # Init jacobians
        self.scheme.J = self.jacob(self.mesh)
        # Init edge jacobians
        # One for each direction
        self.scheme.eJ = self.jacob1D(self.mesh)

    def jacob(self, mesh):
        x = mesh._x

        # Works only for structured mesh (no rotation, only x-axis
        # and/or y-axis stretch)
        # self.jac = J^{-1}
        return (4.0 / (self.dx * self.dy)) * np.ones(x[1:, :-1].shape)

    def jacob1D(self, mesh):
        return mesh.cells.surfaces / 2
