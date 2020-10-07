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

from typing import Sequence, TYPE_CHECKING

from josie.solver.problem import Problem
from josie.solver.state import State

if TYPE_CHECKING:
    from josie.mesh.cellset import CellSet, MeshCellSet


class Scheme(abc.ABC):
    r""" An abstract class representing a scheme to be used during a simulation.

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

    def __init__(self, problem: Problem):
        self.problem = problem

    def accumulate(self, cells: MeshCellSet, neighs: CellSet):
        r""" This method implements the accumulation of all fluxes between
        each cell and its neigbhour. It modifies in place
        :attr:`_fluxes`


        Potentially if the :attr:`problem` is a full problem featuring all
        the terms, this method accumulates the terms

        Attributes
        ----------
        values
            The values of the state fields in each cell

        neighs
            A :class:`CellSet` containing data of neighbour cells corresponding
            to the :attr:`values`

        .. math::

            \numSpaceTerms


        """

        pass

    @abc.abstractmethod
    def update(self, cells: MeshCellSet, dt: float) -> State:
        r""" This method implements the discretization of the time derivative

        .. math::

            \numTimeFull

        Parameters
        ---------
        mesh
            The :class:`Mesh` object used to retrieve the right
            :math:`\text{d}x`

        dt
            Time step

        Returns
        -------
        The term :math:`\numTime`

        """

        raise NotImplementedError

    @abc.abstractmethod
    def CFL(
        self,
        cells: MeshCellSet,
        CFL_value: float,
    ) -> float:
        r""" This method returns the optimal `dt` value that fulfills the CFL
        condition for the concrete the given scheme

        Parameters
        ----------
        cells:
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

    def post_init(self, cells: MeshCellSet, neighbours: Sequence[CellSet]):
        r""":class:`Scheme` can implement a :meth:`post_init` in order to
        perform operations after the :meth:`Solver.init` initialize the
        solver state

        Can be used to store additional data, for example, to compute the
        CFL in an optimized way

        Parameters
        ----------
        values
            A :math:`N_x \times N_y \times N_\text{eqs
        """

        # Initialize the datastructure containing the fluxes
        self._fluxes: State = np.empty_like(cells.values)

    def pre_step(self, cells: MeshCellSet, neighbours: Sequence[CellSet]):
        """
        Hook called just before the fluxes accumulation. It's used by default
        to reset the fluxes array to zeros. It can be extended to do other
        things like for :class:`DiffusiveScheme`.

        Parameters
        ----------
        cells
            A :class:`MeshCellSet` containing the cell data at the current time
        neighbours
            An iterable of :class:`CellSet` containing all the sets of
            neighbours of the mesh cells. For example in a 2D structured cell,
            each cell is gonna have a left, right, top and bottom
            :class:`CellSet`
        """

        self._fluxes.fill(0)

    def post_step(self, cells: MeshCellSet, neighbours: Sequence[CellSet]):
        r""":class:`Scheme` can implement a post-step hook that is executed by the
        solver after the update step.
        It can be needed, for example, to apply an :class:`~.euler.eos.EOS`

        cells
            A :class:`MeshCellSet` containing the cell data at the current time
        neighbours
            An iterable of :class:`CellSet` containing all the sets of
            neighbours of the mesh cells. For example in a 2D structured cell,
            each cell is gonna have a left, right, top and bottom
            :class:`CellSet`
        """

        pass
