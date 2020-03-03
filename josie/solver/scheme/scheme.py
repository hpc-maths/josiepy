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

from .state import State


class Scheme(metaclass=abc.ABCMeta):
    r""" An abstract class representing a scheme to be used during a simulation.

    A general problem can be written in a compact way:

    ..math::

    \pdv{\vb{q}}{t} + \div{\vb{F\qty(\vb{q})}} + \vb{B}\qty(\vb{q}) \cdot
        \gradient{\vb{q}} = \vb{s\qty(\vb{q})}

    This class provides implementation of the discrete numerical schemes for
    the terms :math:`\div{\vb{F\qty(\vb{q})}}, \vb{B}\qty(\vb{q}) \cdot
        \gradient{\vb{q}} `
    """

    @abc.abstractmethod
    def convective_flux(
        self,
        values: State,
        neigh_values: State,
        normals: np.ndarray,
        surfaces: np.ndarray,
    ):
        r""" This is the convective flux implementation of the scheme. See
        :cite:`toro` for a great overview on numerical methods for hyperbolic
        problems.
        This method implements a scheme as a 1D scheme operating on a cell and
        its neighbour (i.e. the :math:`\mathcal{F}` function in the following
        equation)

        ..math:

        \mathbf{U}_i^{k+1} = \mathbf{U}_i^{k} -
            \frac{\text{d}t}{V} \mathcal{F}
            \left(\mathbf{U}_i^{k}, \mathbf{U}_{i+1}^{k} \right)
        """

        raise NotImplementedError

    @abc.abstractmethod
    def CFL(
        self,
        values: State,
        volumes: State,
        normals: np.ndarray,
        surfaces: np.ndarray,
        CFL_value: float,
    ) -> float:
        """ This method returns the optimal `dt` value that fulfills the CFL
        condition for the concrete the given scheme

        Parameters
        ----------
        values
            A :class:`np.ndarray` that has dimension [Nx * Ny * 9] containing
            the values for all the states in all the mesh points
        volumes
            A :class:`np.ndarray` that has the dimensions [Nx * Ny] containing
            the values of the cell volumes
        normals
            A :class:`np.ndarray` that has the dimensions [Nx * Ny * num_points
            * 2] containing the values of the normals to the faces of the cell
        surfaces
            A :class:`np.ndarray` that has the dimensions [Nx * Ny *
            num_points] containing the values of the face surfaces of the face
            connecting the cell to is neighbour
        CFL_value
            The value of the CFL coefficient to impose

        Returns
        -------
        dt
            The Optimal `dt` fulfilling the CFL condition for the given
            CFL number
        """

        raise NotImplementedError

    def post_step(self, values: State) -> State:
        """ Schemes can implement a post-step hook that is executed by the
        solver after the update step.
        It can be needed, for example, to apply an Equation of State
        """

        pass
