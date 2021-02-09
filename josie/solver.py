# josiepy
# Copyright © 2019 Ruben Di Battista
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

from josie.state import State
from josie.scheme import Scheme


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
        self.t = 0

        # Init data structure for field values
        self.mesh.cells._values = self.Q.from_mesh(self.mesh)

        # First set all the values for the internal cells
        # The actual values are a view of only the internal cells
        init_fun(self.mesh.cells)

        # Note: Corner values are unused

        self.mesh.create_neighbours()
        self.mesh.update_ghosts(self.t)

        # Initialize the scheme datastructures (notably the fluxes)
        self.scheme.post_init(self.mesh.cells)

    def copy(self) -> Solver:
        """ This methods copies the :class:`Solver` object into another """

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
