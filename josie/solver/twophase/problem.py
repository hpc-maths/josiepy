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
    r""" This returns the tensor representing the flux for a two-fluid model
    as described originally by :cite:`baer_nunziato`

    A general problem can be written in a compact way:

    ..math::

    \pdv{\vb{q}}{t} + \div{\vb{F\qty(\vb{q})}} + \vb{B}\qty(\vb{q}) \cdot
        \gradient{\vb{q}} = \vb{s\qty(\vb{q})}

    This function needs to return :math:`\vb{F}\qty(\vb{q})`


    Parameters
    ----------
    state_array
        A :class:`np.ndarray` that has dimension `[Nx * Ny * 19]` containing
        the values for all the state variables in all the mesh points

    Returns
    ---------
    F
        An array of dimension `[Nx * Ny * 9 * 2]`, i.e. an array that of each
        x cell and y cell stores the 9*2 flux tensor
        The flux tensor is:
        ..math::

        \begin{bmatrix}
            0 & 0
            \alpha_1 \rho u_1 & \alpha_1 \rho v_1 \\
            \alpha_1(\rho_1 u_1^2 + p_1) & \alpha_1 \rho_1 u_1 v_1 \\
            \alpha_1 \rho_1 v_1 u_1 * \alpha_1(\rho v_1^ 2 + p_1) \\
            \alpha_1(\rho_1 E_1 + p_1)u_1 & \alpha_1 (\rho_1 E + p)v_1
            \alpha_2 \rho u_2 & \alpha_2 \rho v_2 \\
            \alpha_2(\rho_2 u_2^2 + p_2) & \alpha_2 \rho_2 u_2 v_2 \\
            \alpha_2 \rho_2 v_2 u_2 * \alpha_2(\rho v_2^ 2 + p_2) \\
            \alpha_2(\rho_2 E_2 + p_2)u_2 & \alpha_2 (\rho_2 E + p)v_2
        \end{bmatrix}
    """

    num_cells_x, num_cells_y, _ = state_array.shape

    # Flux tensor
    F = np.zeros((num_cells_x, num_cells_y, 9, 2))

    rhoU1 = state_array[:, :, Q.fields.rhoU1]
    rhoV1 = state_array[:, :, Q.fields.rhoV1]
    rhoE1 = state_array[:, :, Q.fields.rhoE1]
    U1 = state_array[:, :, Q.fields.U1]
    V1 = state_array[:, :, Q.fields.V1]
    p1 = state_array[:, :, Q.fields.p1]

    rhoUU1 = np.multiply(rhoU1, U1)
    rhoUV1 = np.multiply(rhoU1, V1)
    rhoVV1 = np.multiply(rhoV1, V1)
    rhoVU1 = np.multiply(rhoV1, U1)

    rhoU2 = state_array[:, :, Q.fields.rhoU1]
    rhoV2 = state_array[:, :, Q.fields.rhoV1]
    rhoE2 = state_array[:, :, Q.fields.rhoE1]
    U2 = state_array[:, :, Q.fields.U1]
    V2 = state_array[:, :, Q.fields.V1]
    p2 = state_array[:, :, Q.fields.p1]

    rhoUU2 = np.multiply(rhoU2, U2)
    rhoUV2 = np.multiply(rhoU2, V2)
    rhoVV2 = np.multiply(rhoV2, V2)
    rhoVU2 = np.multiply(rhoV2, U2)

    F[:, :, 1, 0] = rhoU1
    F[:, :, 1, 1] = rhoV1
    F[:, :, 2, 0] = rhoUU1 + p1
    F[:, :, 2, 1] = rhoUV1
    F[:, :, 3, 0] = rhoVU1
    F[:, :, 3, 1] = rhoVV1 + p1
    F[:, :, 4, 0] = np.multiply(rhoE1 + p1, U1)
    F[:, :, 4, 1] = np.multiply(rhoE1 + p1, V1)

    F[:, :, 1, 0] = rhoU2
    F[:, :, 1, 1] = rhoV2
    F[:, :, 2, 0] = rhoUU2 + p2
    F[:, :, 2, 1] = rhoUV2
    F[:, :, 3, 0] = rhoVU2
    F[:, :, 3, 1] = rhoVV2 + p2
    F[:, :, 4, 0] = np.multiply(rhoE2 + p2, U2)
    F[:, :, 4, 1] = np.multiply(rhoE2 + p2, V2)

    return F
