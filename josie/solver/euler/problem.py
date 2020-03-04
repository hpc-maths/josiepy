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

from .state import Q


def flux(state_array: Q) -> np.ndarray:
    r""" This returns the tensor representing the flux for an Euler model

    A general problem can be written in a compact way:

    .. math::

        \pdv{\vb{q}}{t} + \div{\vb{F\qty(\vb{q})}} + \vb{B}\qty(\vb{q}) \cdot
            \gradient{\vb{q}} = \vb{s\qty(\vb{q})}

    This function needs to return :math:`\vb{F}\qty(\vb{q})`

    Parameters
    ----------
    state_array
        A :class:`np.ndarray` that has dimension :math:`Nx \times Ny \times 9`
        containing the values for all the state variables in all the mesh
        points

    Returns
    ---------
    F
        An array of dimension :math:`Nx \times Ny \times 4 \times 2`, i.e. an
        array that of each :math:`x` cell and :math:`y` cell stores the
        :math:`4 \times 2` flux
        tensor

        The flux tensor is:

        .. math::

            \begin{bmatrix}
                \rho u & \rho v \\
                \rho u^2 + p & \rho uv \\
                \rho vu & \rho v^ 2 + p \\
                (\rho E + p)U & (\rho E + p)V
            \end{bmatrix}
    """

    num_cells_x, num_cells_y, _ = state_array.shape

    # Flux tensor
    F = np.empty((num_cells_x, num_cells_y, 4, 2))

    rhoU = state_array[:, :, Q.fields.rhoU]
    rhoV = state_array[:, :, Q.fields.rhoV]
    rhoE = state_array[:, :, Q.fields.rhoE]
    U = state_array[:, :, Q.fields.U]
    V = state_array[:, :, Q.fields.V]
    p = state_array[:, :, Q.fields.p]

    rhoUU = np.multiply(rhoU, U)
    rhoUV = np.multiply(rhoU, V)
    rhoVV = np.multiply(rhoV, V)
    rhoVU = np.multiply(rhoV, U)

    F[:, :, 0, 0] = rhoU
    F[:, :, 0, 1] = rhoV
    F[:, :, 1, 0] = rhoUU + p
    F[:, :, 1, 1] = rhoUV
    F[:, :, 2, 0] = rhoVU
    F[:, :, 2, 1] = rhoVV + p
    F[:, :, 3, 0] = np.multiply(rhoE + p, U)
    F[:, :, 3, 1] = np.multiply(rhoE + p, V)

    return F
