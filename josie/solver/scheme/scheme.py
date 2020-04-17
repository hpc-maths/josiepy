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
import abc

import numpy as np

from josie.solver.state import State
from josie.solver.problem import Problem
from josie.mesh.mesh import Mesh


class Scheme(metaclass=abc.ABCMeta):
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

    def accumulate_convective(
        self,
        values: State,
        neigh_values: State,
        normals: np.ndarray,
        surfaces: np.ndarray,
    ) -> State:
        r""" This method implements the accumulation for the convective
        fluxes between each cell and its neighbour.

        .. math::

            \numConvectiveFaces
        """

        return np.zeros_like(values)

    def accumulate_nonconservative(
        self,
        values: State,
        neigh_values: State,
        normals: np.ndarray,
        surfaces: np.ndarray,
    ) -> State:
        r""" This method implements the accumulation for the non-conservative
        fluxes between each cell and its neighbour.

        .. math::

            \numNonConservativeFaces
        """

        return np.zeros_like(values)

    def accumulate_source(
        self,
        values: State,
        neigh_values: State,
        normals: np.ndarray,
        surfaces: np.ndarray,
    ) -> State:
        r""" This method implements the accumulation for the source
        terms between each cell and its neighbour.

        .. math::

            \numSource
        """

        return np.zeros_like(values)

    def accumulate(
        self,
        values: State,
        neigh_values: State,
        normals: np.ndarray,
        surfaces: np.ndarray,
    ) -> State:
        r""" This method implements the accumulation of all fluxes between
        each cell and its neigbhour

        Potentially if the :attr:`problem` is a full problem featuring all
        the terms, this method accumulates the terms :math:`\numSpaceTerms`
        """
        fluxes = np.zeros_like(values)

        for accumulate_fun in [
            self.accumulate_convective,
            self.accumulate_nonconservative,
            self.accumulate_source,
        ]:
            fluxes += accumulate_fun(values, neigh_values, normals, surfaces)

        return fluxes

    @abc.abstractmethod
    def update(self, fluxes: State, mesh: Mesh, dt: float) -> State:
        r""" This method implements the discretization of the time derivative

        .. math::

            \numTimeFull

        Returns
        -------
        The term :math:`\numTime`

        """

        raise NotImplementedError

    @abc.abstractmethod
    def CFL(
        self,
        values: State,
        volumes: State,
        surfaces: np.ndarray,
        CFL_value: float,
    ) -> float:
        r""" This method returns the optimal `dt` value that fulfills the CFL
        condition for the concrete the given scheme

        Parameters
        ----------
        values
            A :class:`np.ndarray` that has dimension :math:`Nx \times Ny \times
            N_\text{fields}` containing the values for all the states in all
            the mesh points
        volumes
            A :class:`np.ndarray` that has the dimensions :math:`Nx \times Ny`
            containing the values of the cell volumes
        normals
            A :class:`np.ndarray` that has the dimensions :math:`Nx \times Ny
            \times N_\text{centroids} \times 2` containing the values of the
            normals to the faces of the cell
        surfaces
            A :class:`np.ndarray` that has the dimensions :math:`Nx \times Ny
            \times N_\text{centroids}` containing the values of the face
            surfaces of the face connecting the cell to is neighbour
        CFL_value
            The value of the CFL coefficient to impose

        Returns
        -------
        The Optimal `dt` fulfilling the CFL condition for the given
        CFL number
        """

        raise NotImplementedError

    def post_init(self, values: State):
        """ :class:`Scheme` can implement a :meth:`post_init` in order to
        perform operations after the :meth:`Solver.init` initialize the
        solver state

        Can be used to store additional data, for example, to compute the
        CFL in an optimized way

        Parameters
        ----------
        values
            The values of the fields in the mesh cells
        """

        pass

    def post_step(self, values: State) -> State:
        """ :class:`Scheme` can implement a post-step hook that is executed by the
        solver after the update step.
        It can be needed, for example, to apply an Equation of State
        """

        pass
