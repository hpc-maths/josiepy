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
r"""
We create one big state that contains the actual conservative variables that
are used in the flux together with the "auxiliary" variables that are instead
needed, for example, to compute the speed of sound.

* `rho`: density :math:`\rho`
* `rhoU`: component along :math:`x` of the velocity :math:`u`, multiplied by
   the density
* `rhoV`: component along :math:`y` of the velocity :math:`v`, mvltiplied by
   the density
* `rhoE`: total energy multiplied by the density :math:`\rho E`
* `rhoe`: internal energy multiplied by the density :math:`\rhoe`
* `U`: component along :math:`x` of the velocity :math:`u`
* `V`: component along :math:`y` of the velocity :math:`v`
* `p`: pressure :math:`p`
* `c`: sound velocity :math:`c`
"""
from __future__ import annotations

from enum import IntEnum

from josie.solver.state import State


class Fields(IntEnum):
    rho = 0
    rhoU = 1
    rhoV = 2
    rhoE = 3
    rhoe = 4
    U = 5
    V = 6
    p = 7
    c = 8


class Q(State):
    fields = Fields

    def conservative(self) -> Q:
        """ Returns the conservative part of the state """

        return self[..., :4]
