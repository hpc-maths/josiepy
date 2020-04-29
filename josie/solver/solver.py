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

import abc
import numpy as np
import os

from dataclasses import dataclass
from typing import (
    Callable,
    List,
    NoReturn,
    Union,
    Tuple,
    Type,
    TYPE_CHECKING,
)

from .state import State
from .scheme import Scheme

from josie.mesh import NormalDirection

if TYPE_CHECKING:
    from josie.mesh import Mesh


@dataclass
class Neighbour:
    """ A :class:`namedtuple` representing the set of neighbours to some values.
    It ships the values of the fields in the neighbour cells, together with the
    face normals and the face surfaces"""

    values: State
    normals: np.ndarray
    surfaces: np.ndarray


class Solver(metaclass=abc.ABCMeta):

    # Type Checking
    _values: State
    _neighs: Tuple[Neighbour, ...]

    def __init__(self, mesh: Mesh, Q: Type[State], scheme: Scheme):
        """ This class is used to solve a problem governed by PDEs.

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
        values
            An array of dimensions (num_cells_x, num_cells_y, state_size)
            storing the value of the :class:`State` for each cell of the
            :class:`Mesh`
        left_ghost
            An array of dimensions (num_cells_y, state_size) storing the values
            of the :class:`State` on the ghost cells that are used to apply
            the :class:`BoundaryCondition` for the left boundary
        btm_ghost
            An array of dimensions (num_cells_y, state_size) storing the values
            of the :class:`State` on the ghost cells that are used to apply
            the :class:`BoundaryCondition` for the bottom boundary
        right_ghost
            An array of dimensions (num_cells_y, state_size) storing the values
            of the :class:`State` on the ghost cells that are used to apply
            the :class:`BoundaryCondition` for the right boundary
        top_ghost
            An array of dimensions (num_cells_y, state_size) storing the values
            of the :class:`State` on the ghost cells that are used to apply
            the :class:`BoundaryCondition` for the top boundary
        """

        self.mesh = mesh
        self.Q = Q
        self.scheme = scheme

    # In order to apply BC, we create a view of the self._values array  per
    # each side of the domain storing the State values of the ghost cells.
    @property
    def left_ghost(self) -> np.ndarray:
        return self._values[0, 1:-1]

    @left_ghost.setter
    def left_ghost(self, value: np.ndarray):
        self._values[0, 1:-1, :] = value

    @property
    def right_ghost(self) -> np.ndarray:
        return self._values[-1, 1:-1]

    @right_ghost.setter
    def right_ghost(self, value: np.ndarray):
        self._values[-1, 1:-1, :] = value

    @property
    def top_ghost(self) -> np.ndarray:
        if self.mesh.oneD:
            raise AttributeError("Mesh is 1D")

        return self._values[1:-1, -1]

    @top_ghost.setter
    def top_ghost(self, value: np.ndarray):
        if self.mesh.oneD:
            raise AttributeError("Mesh is 1D")

        self._values[1:-1, -1, :] = value

    @property
    def btm_ghost(self) -> np.ndarray:
        if self.mesh.oneD:
            raise AttributeError("Mesh is 1D")

        return self._values[1:-1, 0]

    @btm_ghost.setter
    def btm_ghost(self, value: np.ndarray):
        if self.mesh.oneD:
            raise AttributeError("Mesh is 1D")

        self._values[1:-1, 0] = value

    @property
    def values(self) -> np.ndarray:
        return self._values[1:-1, 1:-1]

    @values.setter
    def values(self, value: np.ndarray):
        self._values[1:-1, 1:-1, :] = value

    @property
    def neighbours(self) -> Tuple[Neighbour, ...]:
        """ A property returning an iterable of neighbours of the
        :attr:`values`

        Returns
        -------
        neighbours
            A tuple of :class:`Neighbour`.
        """

        try:
            return self._neighs
        except AttributeError:
            neighs: List[Neighbour] = []

            left: Neighbour = Neighbour(
                values=self._values[:-2, 1:-1],
                normals=self.mesh.normals[..., NormalDirection.LEFT, :],
                surfaces=self.mesh.surfaces[..., NormalDirection.LEFT],
            )

            right: Neighbour = Neighbour(
                values=self._values[2:, 1:-1],
                normals=self.mesh.normals[..., NormalDirection.RIGHT, :],
                surfaces=self.mesh.surfaces[..., NormalDirection.RIGHT],
            )

            neighs.extend((left, right))

            if not (self.mesh.oneD):
                top: Neighbour = Neighbour(
                    values=self._values[1:-1, 2:],
                    normals=self.mesh.normals[..., NormalDirection.TOP, :],
                    surfaces=self.mesh.surfaces[..., NormalDirection.TOP],
                )
                btm: Neighbour = Neighbour(
                    values=self._values[1:-1, :-2],
                    normals=self.mesh.normals[..., NormalDirection.BOTTOM, :],
                    surfaces=self.mesh.surfaces[..., NormalDirection.BOTTOM],
                )

                neighs.extend((top, btm))

            self._neighs = tuple(neighs)
            return self._neighs

    def init(self, init_fun: Callable[[Solver], NoReturn]):
        """
        This method initialize the internal values of the cells of the
        :class:`~.Mesh` and the values of the ghost cells that apply the
        :class:`~.BoundaryCondition` for each boundary of the domain

        **Data Structure**

        The data structure holding the values of the field on the mesh is a
        :class:`numpy.ndarray` of dimensions (``num_cells_x+2``,
        ``num_cells_y+2``, ``state_size``).  That is, we have one layer of
        ghosts per direction.  (The corner cells are unused)

        Parameters
        ---------
        init_fun
            The function to use to initialize the value in the domain

        """

        # Init data structure
        self._values = self.Q.from_mesh(self.mesh)

        # First set all the values for the internal cells
        # The actual values are a view of only the internal cells
        init_fun(self)

        # Corner values are unused, with set to NaN
        self._values[0, 0] = np.nan
        self._values[0, -1] = np.nan
        self._values[-1, -1] = np.nan
        self._values[-1, 0] = np.nan

        self.scheme.post_init(self.values)

        self._update_ghosts()

    def _update_ghosts(self):
        """ This method updates the ghost cells of the mesh with the current
        values depending on the specified boundary condition """

        # TODO: Make this a loop over self.mesh.boundaries as we did for
        # neighbours

        # Left BC: Create the left layer of ghost cells
        self.left_ghost = self.mesh.left.bc(
            self, self.mesh.centroids[0, :], self.values[0, :]  # type: ignore
        )  # type: ignore

        # Right BC
        self.right_ghost = self.mesh.right.bc(
            self,
            self.mesh.centroids[-1, :],  # type: ignore
            self.values[-1, :],
        )  # type: ignore

        if not (self.mesh.oneD):
            # Bottom BC
            self.btm_ghost = self.mesh.bottom.bc(
                self,
                self.mesh.centroids[:, 0],  # type: ignore
                self.values[:, 0],
            )  # type: ignore

            # Top BC
            self.top_ghost = self.mesh.top.bc(
                self,
                self.mesh.centroids[:, -1],  # type: ignore
                self.values[:, -1],
            )  # type: ignore

    def step(self, dt: float):
        """ This method advances one step in time (for the moment using an
        explicit Euler scheme for time integration, but in future we will
        provide a way to give arbitrary time schemes)

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

        # We accumulate the delta fluxes for each set of neighbours
        fluxes = np.zeros_like(self.values)

        # Loop on all the cells neigbours
        for neighs in self.neighbours:
            fluxes += self.scheme.accumulate(
                self.values, neighs.values, neighs.normals, neighs.surfaces
            )

        # Update
        self.values -= self.scheme.update(fluxes, self.mesh, dt)

        # Let's put here an handy post step if needed after the values update
        self.scheme.post_step(self.values)

        # Keep ghost cells updated
        self._update_ghosts()

    def save(self, t, filename: os.PathLike):
        """ This method saves the simulation instant in a `xdmf` file
        supporting time series.

        Parameters
        ---------
        t
            The time instant to save
        filename
            The name of the file holding the time series. It's an XDMF file
        """

        raise NotImplementedError

    def plot(self):
        """ Plot the current state of the simulation in a GUI.
        """
        plt = self.mesh.backend
        plt.update(self)

    def animate(self, t):
        """ Animate the simulation. Call :meth:`animate` for each time instant
        you want to provide in the animation.

        Parameters
        ----------
        t
            The time instant to animate
        """
        plt = self.mesh.backend
        plt.append(self, t)

    def show(self, fields: Union[List[str], str]):
        """ Display on screen the given fields

        Parameters
        ---------
        fields
            The fields you want to plot
        """

        plt = self.mesh.backend
        plt.show(fields)
