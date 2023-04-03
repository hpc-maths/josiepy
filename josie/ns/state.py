# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

r"""
We create one big state that contains the actual conservative variables that
are used in the flux together with the "auxiliary" variables that are instead
needed, for example, to compute the speed of sound.

.. math::

    \nsState


* ``rho``: density :math:`\rho`
* | ``rhoU``: component along :math:`x` of the velocity :math:`\vb{u}`,
  | multiplied by the density
* | ``rhoV``: component along :math:`y` of the velocity :math:`\vb{u}`,
  | multiplied by the density
* ``rhoE``: total energy multiplied by the density :math:`\rho E`
* ``rhoe``: internal energy multiplied by the density :math:`\rho e`
* ``U``: component along :math:`x` of the velocity :math:`u`
* ``V``: component along :math:`y` of the velocity :math:`v`
* ``p``: pressure :math:`p`
* ``c``: sound velocity :math:`c`
"""
from __future__ import annotations

from josie.fluid.state import (
    DiffState,
)

from josie.euler.state import EulerState
from josie.euler.fields import EulerFields
from josie.state import SubsetState

from .fields import NSGradientFields


class NSGradientState(SubsetState):
    """A :class:`State` class representing the diffusive state variables
    of the Navier-Stokes system, i.e. the variables whose gradient is required
    in the diffusive term"""

    full_state_fields = EulerFields  # Same as Euler
    fields = NSGradientFields


# We add the diffusive subset to the Euler State
class NSState(EulerState, DiffState):
    diff_state = NSGradientState
