# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

from josie.general.schemes.space import MUSCL_Hancock

from josie.mesh.cellset import MeshCellSet, DimensionPair
import numpy as np
import abc


class MUSCL_Hancock_ratio_limiters(MUSCL_Hancock):
    # beta_L and beta_R set to their limit 1
    beta_L, beta_R = 1.0, 1.0

    @abc.abstractmethod
    def xi(self, r: np.ndarray):
        raise NotImplementedError

    def xi_L(self, r: np.ndarray):
        return 2 * self.beta_L * r / (1 - self.omega + (1 + self.omega) * r)

    def xi_R(self, r: np.ndarray):
        return 2 * self.beta_R / (1 - self.omega + (1 + self.omega) * r)

    def compute_slopes(self, cells: MeshCellSet):
        # Compute intercell slopes for each face with a slope limiter
        # We assume here that all cell sizes are the same

        # Minimal tolerance to avoid 0-slopes
        tol = 1e-15

        # Compute slope for each direction
        for i, dim in enumerate(DimensionPair):
            if i >= cells.dimensionality:
                break
            dir_L = dim.value[0].value
            dir_R = dim.value[1].value
            neigh_L = cells.neighbours[dir_L]
            neigh_R = cells.neighbours[dir_R]

            slope_L: np.ndarray = cells.values - neigh_L.values
            slope_R: np.ndarray = neigh_R.values - cells.values

            # Add a tolerance to avoid null slopes when computing the ratio
            slope_ratio = np.where(np.abs(slope_R) > tol, slope_L, 0) / np.where(
                np.abs(slope_R) > tol, slope_R, 1.0
            )
            # Ratio of slopes is given to xi function
            slope_R = self.xi(slope_ratio) * (
                0.5 * (1 + self.omega) * slope_L + 0.5 * (1 - self.omega) * slope_R
            )

            self.slopes[..., dir_L] = -slope_R
            self.slopes[..., dir_R] = slope_R


class MUSCL_Hancock_Superbee_r(MUSCL_Hancock_ratio_limiters):
    def xi(self, r: np.ndarray):
        xi_value = np.zeros_like(r)

        np.copyto(xi_value, 2 * r, where=(r > 0) * (r <= 0.5))
        np.copyto(xi_value, 1, where=(r > 0.5) * (r <= 1))
        np.copyto(
            xi_value,
            np.stack([r, self.xi_R(r), 2 * np.ones_like(r)]).min(axis=0),
            where=(r > 1),
        )

        return xi_value


class MUSCL_Hancock_van_Leer(MUSCL_Hancock_ratio_limiters):
    def xi(self, r: np.ndarray):
        xi_value = np.zeros_like(r)

        np.copyto(
            xi_value,
            np.stack([2 * r / (1 + r), self.xi_R(r)]).min(axis=0),
            where=(r >= 0),
        )

        return xi_value


class MUSCL_Hancock_van_Albada(MUSCL_Hancock_ratio_limiters):
    def xi(self, r: np.ndarray):
        xi_value = np.zeros_like(r)

        np.copyto(
            xi_value,
            np.stack([r * (1 + r) / (1 + r * r), self.xi_R(r)]).min(axis=0),
            where=(r >= 0),
        )

        return xi_value


class MUSCL_Hancock_Minbee(MUSCL_Hancock_ratio_limiters):
    def xi(self, r: np.ndarray):
        xi_value = np.zeros_like(r)

        np.copyto(xi_value, r, where=(r > 0) * (r <= 1.0))
        np.copyto(
            xi_value,
            np.stack([np.ones_like(r), self.xi_R(r)]).min(axis=0),
            where=(r >= 1.0),
        )

        return xi_value
