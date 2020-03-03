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

from josie.solver.scheme import Scheme

from josie.solver.euler import EulerScheme

from .state import Q, Phases


class TwoPhaseScheme(Scheme):
    """ A general base class for two-phase schemes. A two-phase scheme is
    built applying a :class:`EulerScheme` for each partition of the state
    associated to one phase

    Attributes
    ---------
    euler_scheme
        The :class:`EulerScheme` to be applied to each phase system
    """

    def __init__(self, euler_scheme: EulerScheme):
        self.euler_scheme = euler_scheme

    def convective_flux(
        self, values: Q, neigh_values: Q, normals: Q, surfaces: Q
    ):
        """ This schemes implements the Rusanov scheme. See :cite: `rusanov`
        for a detailed view on compressible schemes

        Parameters
        ----------
        values
            A :class:`np.ndarray` that has dimension [Nx * Ny * 19] containing
            the values for all the states in all the mesh points
        neigh_values
            A :class:`np.ndarray` that has the same dimension of `values`. It
            contains the corresponding neighbour values of the state stored in
            `values`, i.e. the neighbour of `values[i]` is `neigh_values[i]`
        normals
            A :class:`np.ndarray` that has the dimensions [Nx * Ny * 2]
            containing the values of the normals to the face connecting the
            cell to its neighbour
        surfaces
            A :class:`np.ndarray` that has the dimensions [Nx * Ny] containing
            the values of the face surfaces of the face connecting the cell to
            is neighbour
        """
        FS = np.empty_like(values).view(Q)

        # We apply the Euler scheme per each phase
        for phase in Phases:
            phase_values = values.get_phase(phase)
            phase_neigh_values = neigh_values.get_phase(phase)

            FS.set_phase(
                self.euler_scheme.convective_flux(
                    phase_values, phase_neigh_values, normals, surfaces
                )
            )

        return FS

    def CFL(
        self,
        values: np.ndarray,
        volumes: np.ndarray,
        normals: np.ndarray,
        surfaces: np.ndarray,
        CFL_value,
    ) -> float:

        # We apply the Euler CFL method per each phase and we take the minimum
        # dt
        dt = 1e9  # Use a big value to initialize
        for phase in Phases:
            phase_values = values.get_phase(phase)
            dt = np.min(
                self.euler_scheme.CFL(
                    phase_values, volumes, normals, surfaces, CFL_value
                ),
                dt,
            )

        return dt
