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

from typing import TYPE_CHECKING

from josie.solver import Solver

from .eos import EOS

if TYPE_CHECKING:
    from josie.mesh import Mesh


class EulerSolver(Solver):
    """ This class accepts as input also the EOS """
    # TODO: Add CFL handling

    def __init__(self, mesh: 'Mesh', eos: EOS):
        self.eos = eos

        super().__init__(mesh)

    def post_step(self):
        for cell in self.mesh.cells.ravel():
            Q = cell.value
            rho = Q.rho
            U = Q.rhoU/rho
            V = Q.rhoV/rho
            rhoe = Q.rhoE - 0.5*rho*(U**2 + V**2)
            e = rhoe/rho
            p = self.eos.p_from_rho_e(rho, e)
            c = self.eos.sound_velocity(rho, p)
            cell.value[4:] = np.array([rhoe, U, V, p, c])
