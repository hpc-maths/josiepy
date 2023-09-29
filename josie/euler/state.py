# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

r"""
We create one big state that contains the actual conservative variables that
are used in the flux together with the "auxiliary" variables that are instead
needed, for example, to compute the speed of sound.

.. math::

    \eulerState


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

from josie.fluid.state import SingleFluidState
from josie.state import SubsetState

from .fields import EulerFields, ConsFields


class EulerConsState(SubsetState):
    """A :class:`State` class representing the conservative state variables
    of the Euler system"""

    full_state_fields = EulerFields
    fields = ConsFields


class EulerState(SingleFluidState):
    r"""The class representing the state variables of the Euler system

    .. math::

        \eulerState

    """
    fields = EulerFields
    cons_state = EulerConsState
    prim_state = EulerConsState
