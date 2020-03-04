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

from .scheme import Scheme
from josie.solver.state import State


class ConvectiveScheme(Scheme):
    r"""
    A mixin that provides the convective scheme implementation.

    A general problem can be written in a compact way:

    ..math::

    \pdv{\vb{q}}{t} + \div{\vb{F\qty(\vb{q})}} + \vb{B}\qty(\vb{q}) \cdot
        \gradient{\vb{q}} = \vb{s\qty(\vb{q})}

    A concrete instance of this class needs to implement the term:

    ..math::

    \int_{V_i} \div{\vb{F}\qty(\vb{q})} \dd{V} =
        \oint_{\partial V_i} \vb{F}\qty(\vb{q}) \cdot \hat{\vb{n}} \dd{S}
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
