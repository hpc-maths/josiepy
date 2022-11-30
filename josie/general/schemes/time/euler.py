# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

from josie.scheme.time import TimeScheme
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
        self.pre_accumulate(mesh.cells, dt, t)

        # Accumulate the numerical fluxes over all neighbours
        for neighs in mesh.cells.neighbours:
            self.accumulate(mesh.cells, neighs, t)
