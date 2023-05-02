# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

from josie.scheme.convective import ConvectiveScheme

import numpy as np

import abc


class No_Limiter(ConvectiveScheme):
    omega = 0

    def limiter(self, slope_L: np.ndarray, slope_R: np.ndarray):
        # No slope limiter
        return 0.5 * (1 + self.omega) * slope_L + 0.5 * (1 - self.omega) * slope_R


class Beta_limiters(ConvectiveScheme):
    """MUSCL class with a "beta" limiter.
    See Toro, Eleuterio F. Riemann Solvers and Numerical Methods for Fluid
    Dynamics: A Practical Introduction. 3rd ed. Berlin Heidelberg:
    Springer-Verlag, 2009. https://doi.org/10.1007/b79761, page 508"""

    omega = 0

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

    def limiter(self, slope_L: np.ndarray, slope_R: np.ndarray):
        # Compute intercell slopes for each face with a slope limiter
        # We assume here a regular mesh (dx=cst)

        beta = self.beta

        return self.array_max_min_min(
            beta * slope_L,
            slope_R,
            slope_L,
            beta * slope_R,
        ) * (slope_R > 0) + self.array_min_max_max(
            beta * slope_L,
            slope_R,
            slope_L,
            beta * slope_R,
        ) * (
            slope_R < 0
        )


class MinMod(Beta_limiters):
    beta = 1.0


class Superbee(Beta_limiters):
    beta = 2.0


class Ratio_limiters(ConvectiveScheme):
    # beta_L and beta_R set to their limit 1
    beta_L, beta_R = 1.0, 1.0

    omega = 0

    @abc.abstractmethod
    def xi(self, r: np.ndarray):
        raise NotImplementedError

    def xi_L(self, r: np.ndarray):
        return 2 * self.beta_L * r / (1 - self.omega + (1 + self.omega) * r)

    def xi_R(self, r: np.ndarray):
        return 2 * self.beta_R / (1 - self.omega + (1 + self.omega) * r)

    def limiter(self, slope_L: np.ndarray, slope_R: np.ndarray):
        # Compute intercell slopes for each face with a slope limiter
        # We assume here that all cell sizes are the same

        # Minimal tolerance to avoid 0-slopes
        tol = 1e-15

        # Add a tolerance to avoid null slopes when computing the ratio
        slope_ratio = np.where(np.abs(slope_R) > tol, slope_L, 0) / np.where(
            np.abs(slope_R) > tol, slope_R, 1.0
        )
        # Ratio of slopes is given to xi function
        return self.xi(slope_ratio) * (
            0.5 * (1 + self.omega) * slope_L + 0.5 * (1 - self.omega) * slope_R
        )


class Superbee_r(Ratio_limiters):
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


class van_Leer(Ratio_limiters):
    def xi(self, r: np.ndarray):
        xi_value = np.zeros_like(r)

        np.copyto(
            xi_value,
            np.stack([2 * r / (1 + r), self.xi_R(r)]).min(axis=0),
            where=(r >= 0),
        )

        return xi_value


class van_Albada(Ratio_limiters):
    def xi(self, r: np.ndarray):
        xi_value = np.zeros_like(r)

        np.copyto(
            xi_value,
            np.stack([r * (1 + r) / (1 + r * r), self.xi_R(r)]).min(axis=0),
            where=(r >= 0),
        )

        return xi_value


class Minbee(Ratio_limiters):
    def xi(self, r: np.ndarray):
        xi_value = np.zeros_like(r)

        np.copyto(xi_value, r, where=(r > 0) * (r <= 1.0))
        np.copyto(
            xi_value,
            np.stack([np.ones_like(r), self.xi_R(r)]).min(axis=0),
            where=(r >= 1.0),
        )

        return xi_value
