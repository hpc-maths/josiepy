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

from typing import TYPE_CHECKING

from josie.solver.problem import Problem
from josie.solver.state import State

if TYPE_CHECKING:
    from josie.mesh.cellset import CellSet, MeshCellSet
    from josie.mesh import Mesh


class Scheme(abc.ABC):
    r"""An abstract class representing a scheme to be used during a simulation.

    A general problem can be written in a compact way:

    .. math::

        \pdeFull


    A concrete instance of this class needs to implement discretization
    strategies for the terms that are present in a specific :class:`Problem`
    within a Finite Volume discretization method

    * .. math::

        \numConvectiveFull

    * .. math::

        \numNonConservativeFull

    * .. math::

        \numDiffusiveFull

    * .. math::

        \numSourceFull

    Together with the time update scheme to solve:

    .. math::

        \numTime



    Attributes
    ----------
    problem
        An instance of :class:`Problem` representing the physical problem that
        this scheme discretizes
    """

    _fluxes: State

    def __init__(self, problem: Problem):
        self.problem = problem

    @classmethod
    def _all_subclasses(cls):
        """A recursive class method to get all the subclasses of this class"""
        return set(cls.__subclasses__()).union(
            [s for c in cls.__subclasses__() for s in c._all_subclasses()]
        )

    @abc.abstractmethod
    def CFL(
        self,
        cells: MeshCellSet,
        CFL_value: float,
    ) -> float:
        r"""This method returns the optimal `dt` value that fulfills the CFL
        condition for the concrete the given scheme

        Parameters
        ----------
        cells
            A :class:`MeshCellSet` containing the cell data at the current time
            step

        CFL_value
            The value of the CFL coefficient to impose

        Returns
        -------
        dt
            The Optimal `dt` fulfilling the CFL condition for the given
            CFL number
        """

        raise NotImplementedError

    @abc.abstractmethod
    def step(
        self,
        mesh: Mesh,
        dt: float,
        t: float,
    ):
        """This method implements the accumulation over all the neighbours of a
        specific cell of the numerical fluxes. It can be overridden by, for
        example, a :class:`TimeScheme` in order to implement multi-step
        accumulation

        Parameters
        ----------
        mesh
            A :class:`Mesh` containing the state of the mesh

        dt
            Time step

        t
            The current time instant of the simulation

        """

        raise NotImplementedError

    @abc.abstractmethod
    def accumulate(self, cells: MeshCellSet, neighs: CellSet, t: float):
        r"""This method implements the accumulation of all fluxes between
        each cell and its neighbour.


        Potentially if the :attr:`problem` is a full problem featuring all
        the terms, this method accumulates the terms

        .. math::

            \numSpaceTerms


        Parameters
        ----------
        cells
            A :class:`MeshCellSet` containing the state of the mesh cells

        neighs
            A :class:`CellSet` containing data of neighbour cells corresponding
            to the :attr:`cells`

        t
            The time instant at which to compute time-dependent terms

        """

        pass

    def update(self, mesh: Mesh, dt: float, t: float):
        r"""This method implements the time step update. It accumulates all the
        numerical fluxes using the :meth:`Scheme.step` method (possibly in
        multiple steps for high-order time schemes). It modifies the
        :class:`Mesh` object in-place.

        .. math::

            \numTimeFull

        Parameters
        ---------
        mesh
            A :class:`Mesh` containing the state of the mesh at the given time
            step

        dt
            Time step

        t
            The current time instant of the simulation

        """

        cells = mesh.cells

        # Use pre_step to zero-out data arrays
        self.pre_step(cells)

        # Accumulate all the fluxes (in multiple steps if required by the time
        # scheme). This modifies self._fluxes in-place
        self.step(mesh, dt, t)

        # Update the cell values
        cells.values -= self._fluxes * dt / cells.volumes[..., np.newaxis]

        # Let's put here an handy post step if needed after the values update
        self.post_step(cells)

        # Keep ghost cells updated
        mesh.update_ghosts(t)

    def post_init(self, cells: MeshCellSet):
        r""":class:`Scheme` can implement a :meth:`post_init` in order to
        perform operations after the :meth:`Solver.init` initialize the
        solver state

        Can be used to store additional data, for example, to compute the
        CFL in an optimized way

        Parameters
        ----------
        cells
            A :class:`MeshCellSet` containing the state of the mesh cells
        """

        # Initialize the datastructure containing the fluxes
        self._fluxes: State = np.empty_like(cells.values)

    def pre_step(self, cells: MeshCellSet):
        """
        Hook called just before the fluxes accumulation.

        It's used by default to reset the fluxes array to zeros. It can be
        extended to do reset other :class:`Scheme`-specific data containers

        Parameters
        ----------
        cells
            A :class:`MeshCellSet` containing the state of the mesh cells
        """

        self._fluxes.fill(0)

    def pre_accumulate(self, cells: MeshCellSet, t: float):
        """ "
        Hook that can be used to do stuff before the accumulation around all
        the cell faces.

        It can be used for exemple to implement schemes that
        just need the information on the cell and not its neighbours or to
        compute the gradient accessing all the neighbours are the same moment
        as done in :class:`LeastSquareGradient`

        Parameters
        ----------
        cells
            A :class:`MeshCellSet` containing the state of the mesh cells

        t
            Time instant
        """
        pass

    def post_step(self, cells: MeshCellSet):
        r""":class:`Scheme` can implement a post-step hook that is executed by the
        solver after the update step.
        It can be needed, for example, to apply an :class:`~.euler.eos.EOS`

        Attributes
        ----------

        cells
            A :class:`MeshCellSet` containing the state of the mesh cells
        """

        pass
