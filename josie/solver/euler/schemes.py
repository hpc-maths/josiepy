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

import numpy as np

from josie.mesh.cell import Cell, NeighbourCell

from .state import Q as Q_factory
from .problem import flux


def rusanov(cell: Cell, neigh: NeighbourCell):
    Q = Q_factory(0, 0, 0, 0, 0, 0, 0, 0, 0)
    # First four variables of the total state are the conservative
    # variables (rho, rhoU, rhoV, rhoE)
    Q_cons = Q[:4]

    Q_cell = cell.value
    Q_cell_cons = Q_cell[:4]

    # Geometry
    norm = neigh.face.normal
    S = neigh.face.surface

    Q_neigh = neigh.value
    Q_neigh_cons = Q_neigh[:4]

    sigma = np.max((
        np.abs(Q_cell.U) + Q_cell.c,
        np.abs(Q_neigh.U) + Q_neigh.c
    ))

    # Rusanov scheme here
    F = 0.5*(flux(Q_cell) + flux(Q_neigh)).dot(norm) - \
        0.5*sigma*(Q_neigh_cons - Q_cell_cons)

    Q_cons = Q_cons + F*S

    Q[:4] = Q_cons

    return Q
