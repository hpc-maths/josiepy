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

from josie.general.schemes.space import MUSCL_Hancock

from josie.mesh.cellset import MeshCellSet
from josie.mesh.cellset import DimensionPair
import numpy as np
import abc


class MUSCL_Hancock_no_limiter(MUSCL_Hancock):
    """MUSCL class without limiter i.e. slopes are linear extrapolation
    from neigbouring cells."""

    def compute_slopes(self, cells: MeshCellSet):
        # Compute intercell slopes for each face
        # We assume here that all cell sizes are the same
        for i, dim in enumerate(DimensionPair):
            if i >= cells.dimensionality:
                break
            dir_L = dim.value[0].value
            dir_R = dim.value[1].value
            neigh_L = cells.neighbours[dir_L]
            neigh_R = cells.neighbours[dir_R]

            slope_L = cells.values - neigh_L.values
            slope_R = neigh_R.values - cells.values

            # Without limiters
            self.slopes[..., dir_R] = (
                0.5 * (1 + self.omega) * slope_L
                + 0.5 * (1 - self.omega) * slope_R
            )

            self.slopes[..., dir_L] = -self.slopes[..., dir_R]


class MUSCL_Hancock_Beta_limiters(MUSCL_Hancock):
    """MUSCL class with a "beta" limiter.
    See Toro, Eleuterio F. Riemann Solvers and Numerical Methods for Fluid
    Dynamics: A Practical Introduction. 3rd ed. Berlin Heidelberg:
    Springer-Verlag, 2009. https://doi.org/10.1007/b79761, page 508"""

    @abc.abstractproperty
    def beta(self):
        pass

    @staticmethod
    def array_max_min_min(
        arr1: np.ndarray,
        arr2: np.ndarray,
        arr3: np.ndarray,
        arr4: np.ndarray,
    ):
        return np.stack(
            [
                np.zeros_like(arr1),
                np.stack([arr1, arr2]).min(axis=0),
                np.stack([arr3, arr4]).min(axis=0),
            ]
        ).max(axis=0)

    @staticmethod
    def array_min_max_max(
        arr1: np.ndarray,
        arr2: np.ndarray,
        arr3: np.ndarray,
        arr4: np.ndarray,
    ):
        return np.stack(
            [
                np.zeros_like(arr1),
                np.stack([arr1, arr2]).max(axis=0),
                np.stack([arr3, arr4]).max(axis=0),
            ]
        ).min(axis=0)

    def compute_slopes(self, cells: MeshCellSet):
        # Compute intercell slopes for each face with a slope limiter
        # We assume here a regular mesh (dx=cst)

        beta = self.beta

        # Compute slope for each direction
        for i, dim in enumerate(DimensionPair):
            if i >= cells.dimensionality:
                break
            dir_L = dim.value[0].value
            dir_R = dim.value[1].value
            neigh_L = cells.neighbours[dir_L]
            neigh_R = cells.neighbours[dir_R]

            slope_L = cells.values - neigh_L.values
            slope_R = neigh_R.values - cells.values

            slope = (
                self.array_max_min_min(
                    beta * slope_L,
                    slope_R,
                    slope_L,
                    beta * slope_R,
                )
                * (slope_R > 0)
                + self.array_min_max_max(
                    beta * slope_L,
                    slope_R,
                    slope_L,
                    beta * slope_R,
                )
                * (slope_R < 0)
            )

            self.slopes[..., dir_L] = -slope
            self.slopes[..., dir_R] = slope


class MUSCL_Hancock_MinMod(MUSCL_Hancock_Beta_limiters):

    beta = 1.0


class MUSCL_Hancock_SuperBee(MUSCL_Hancock_Beta_limiters):

    beta = 2.0
