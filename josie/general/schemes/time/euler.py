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
from josie.solver.scheme.time import TimeScheme
from josie.mesh import Mesh


class ExplicitEuler(TimeScheme):
    r"""Implements the explicit euler scheme

    .. math::

        \pdeState^{k+1} = \pdeState^k +
            \Delta t \; \vb{f}\qty(t; \pdeState^k,\pdeGradient^k)
    """

    time_order: float = 1

    def step(self, mesh: Mesh, dt: float, t: float):
        """For :class:`ExplicitEuler`, we just accumulate once the fluxes at
        time :math:`t = t_k`. So nothing to do.
        """
        # Do the pre_accumulate
        self.pre_accumulate(mesh.cells, t)

        # Accumulate the numerical fluxes over all neighbours
        for neighs in mesh.cells.neighbours:
            self.accumulate(mesh.cells, neighs, t)
