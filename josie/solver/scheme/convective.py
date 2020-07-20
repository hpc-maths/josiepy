# josiepy
# Copyright © 2020 Ruben Di Battista
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

from .scheme import Scheme


class ConvectiveScheme(Scheme):
    """ A mixin that provides the scheme implementation for the convective
    term"""

    @abc.abstractmethod
    def F(
        self,
        values: State,
        neigh_values: State,
        normals: np.ndarray,
        surfaces: np.ndarray,
    ) -> State:
        r""" This is the convective flux implementation of the scheme. See
        :cite:`toro_riemann_2009` for a great overview on numerical methods for
        hyperbolic problems.

        A general problem can be written in a compact way:

        .. math::

            \pdeFull


        A concrete instance of this class needs to implement the discretization
        of the numerical flux on **one** face of a cell. It needs to implement
        the term :math:`\numConvective`

        .. math::

            \numConvectiveFull


        Parameters
        ----------
        values
            The values of the state fields in each cell

        neigh_values
            The values of the state fields in the each neighbour of a cell

        normals
            The normal unit vectors associated to the face between each cell
            and its neigbour

        surfaces
            The surface values associated at the face between each cell and its
            neighbour

        Returns
        -------
        F
            The value of the numerical convective flux multiplied by the
            surface value :math:`\numConvective`

        """
        raise NotImplementedError

    def accumulate_convective(
        self,
        values: State,
        neigh_values: State,
        normals: np.ndarray,
        surfaces: np.ndarray,
    ):

        # Compute fluxes computed eventually by the other schemes (e.g.
        # nonconservative)
        super().accumulate_convective(values, neigh_values, normals, surfaces)

        # Add conservative contribution
        self._fluxes += self.F(values, neigh_values, normals, surfaces)
