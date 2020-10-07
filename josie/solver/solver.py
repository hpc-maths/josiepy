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

import numpy as np

from typing import (
    Callable,
    Sequence,
    List,
    NoReturn,
    Union,
    Type,
    TYPE_CHECKING,
)

from josie.mesh.cellset import CellSet, MeshCellSet, NeighbourDirection

from .state import State
from .scheme import Scheme


if TYPE_CHECKING:
    from josie.mesh.mesh import Mesh


class Solver:
    r""" This class is used to solve a problem governed by PDEs.

    The internal state of the mesh is stored in :attr:`values`, while the
    values of the ghost cells (used to apply the
    :class:`BoundaryCondition`) are stored respectively in
    :attr:`left_ghost`, :attr:`btm_ghost`, :attr:`right_ghost`,
    :attr:`top_ghost`. They're all numpy arrays or views to numpy arrays.

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

    @property
    def neighbours(self) -> Sequence[CellSet]:
        """A property returning an iterable of neighbours of the
        :attr:`values`

        Returns
        -------
        neighbours
            A tuple of :class:`CellSet`.
        """

        # OPTIMIZE: Probably instead of the Iterable using a numpy array
        # would help to make computations faster

        try:
            return self._neighs
        except AttributeError:
            cells = self.mesh.cells
            neighs = [
                cells.get_neighbours(NeighbourDirection.LEFT),
                cells.get_neighbours(NeighbourDirection.RIGHT),
            ]

            if not (self.mesh.dimensionality == 1):
                neighs.extend(
                    (
                        cells.get_neighbours(NeighbourDirection.TOP),
                        cells.get_neighbours(NeighbourDirection.BOTTOM),
                    )
                )

            self._neighs = tuple(neighs)
        return self._neighs

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

        # Corner values are unused, set to NaN
        self.mesh.cells._values[0, 0] = np.nan
        self.mesh.cells._values[0, -1] = np.nan
        self.mesh.cells._values[-1, -1] = np.nan
        self.mesh.cells._values[-1, 0] = np.nan

        self._update_ghosts()

        # Initialize the scheme datastructures (notably the fluxes)
        self.scheme.post_init(self.mesh.cells, self.neighbours)

    def _update_ghosts(self):
        """This method updates the ghost cells of the mesh with the current
        values depending on the specified boundary condition"""

        for boundary in self.mesh.boundaries:
            boundary.bc(self)

    def step(self, dt: float):
        """This method advances one step in time using the
        :meth:`Scheme.update` method of the given numerical scheme.

        A `scheme` callable gets as input the internal values of the cells, the
        neighbour values, the normals associated to the neighbours and the
        value of the face surfaces. A `scheme` generally has to be coded as a
        1D scheme that operates only on the *right* neighbour. It is then
        called (in 2D) 4 times, one for each set of neighbours (left, bottom,
        right, top).  As an example, when called for the right neighbours, the
        data structures sent to the `scheme` callable for `values` and
        `neigh_values` are:

        **values**

        +-----------------+-----------------+-----------------+
        | ``values[0,2]`` | ``values[1,2]`` | ``values[2,2]`` |
        +-----------------+-----------------+-----------------+
        | ``values[0,1]`` | ``values[1,1]`` | ``values[2,1]`` |
        +-----------------+-----------------+-----------------+
        | ``values[0,0]`` | ``values[1,0]`` | ``values[2,0]`` |
        +-----------------+-----------------+-----------------+

        **neighbours**

        +-----------------+-----------------+--------------------+
        | ``values[1,2]`` | ``values[2,2]`` | ``right_ghost[2]`` |
        +-----------------+-----------------+--------------------+
        | ``values[1,1]`` | ``values[2,1]`` | ``right_ghost[1]`` |
        +-----------------+-----------------+--------------------+
        | ``values[1,0]`` | ``values[2,0]`` | ``right_ghost[0]`` |
        +-----------------+-----------------+--------------------+

        Parameters
        ----------
        dt
            Time increment of the step
        """
        cells = self.mesh.cells

        self.scheme.pre_step(cells, self.neighbours)

        # Loop on all the cells neigbours
        for neighs in self.neighbours:
            self.scheme.accumulate(cells, neighs)

        # Update
        cells.values -= self.scheme.update(cells, dt)

        # Let's put here an handy post step if needed after the values update
        self.scheme.post_step(cells, self.neighbours)

        # Keep ghost cells updated
        self._update_ghosts()

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
