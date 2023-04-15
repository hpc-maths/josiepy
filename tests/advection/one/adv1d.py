# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

""" A small script to solve an advection equation in 1D as a test for josiepy
"""
import numpy as np


def init(x: np.ndarray):
    """Init function. It inits the state with a Rieman problemm"""
    u = np.empty(x.shape)

    u[np.where(x > 0.45)] = 1
    u[np.where(x <= 0.45)] = 0

    return u


def flux(u: np.ndarray, a: np.ndarray):
    return np.multiply(a, u)


def upwind(u: np.ndarray, a: np.ndarray):
    au = np.zeros(u[1:-1].shape)

    # Below vectorized form of this:
    # for i in np.arange(len(a)):
    #     idx_u = i+1
    #     if a[i] >= 0:
    #         au[i] = a[i]*u[idx_u] - a[i-1]*u[idx_u-1]
    #     else:
    #         au[i] = a[i+1]*u[idx_u+1] - a[i]*u[idx_u]

    u_state = u[1:-1]
    a_state = a[1:-1]

    idx_a_plus = np.asarray(a_state >= 0).nonzero()[0]
    idx_a_minus = np.asarray(a_state < 0).nonzero()[0]

    a_plus = a_state[idx_a_plus]
    a_plus_left = a[idx_a_plus]
    a_minus = a_state[idx_a_minus]
    a_minus_right = a[idx_a_minus + 1]

    u_plus = u_state[idx_a_plus]
    u_plus_left = u[idx_a_plus]
    u_minus = u_state[idx_a_minus]
    u_minus_right = u[idx_a_minus + 2]

    au[idx_a_plus] = flux(u_plus, a_plus) - flux(u_plus_left, a_plus_left)
    au[idx_a_minus] = flux(u_minus_right, a_minus_right) - flux(u_minus, a_minus)

    return au


def advection_velocity(t, x, tf):
    f_x = np.ones(x.shape)
    f_t = np.ones(t.shape)

    return np.outer(f_t, f_x)


def main(nx, tf, CFL, plot=False):
    dx = 1 / nx
    x = np.arange(0 + dx / 2, 1, dx)
    dx = x[1] - x[0]

    # Use a temporary dt to get the max of the advection velocity
    time = np.arange(0, tf, dx)
    temp_a = np.abs(np.max(advection_velocity(time, x, tf)))
    dt = CFL * dx / np.abs(temp_a)

    # Recompute the time with CFL
    time = np.arange(0, tf, dt)
    u = init(x)

    # Add periodic BC
    u = np.hstack((u[-1], u, u[0]))
    u_new = np.empty(u.shape)

    # Allocate a
    a = np.empty(u.shape)

    # Storing solution
    solution = np.empty((len(time), len(x)))

    for i, t in enumerate(time):
        # Append to solution array
        solution[i, :] = u[1:-1]

        # Time dependent advection velocity
        a[1:-1] = advection_velocity(t, x, tf)

        # Periodic BC
        a[0] = a[-2]
        a[-1] = a[1]

        # State update
        u_new[1:-1] = u[1:-1] - dt / dx * upwind(u, a)

        # Periodic BC
        u_new[0] = u_new[-2]
        u_new[-1] = u[1]

        u = u_new

    return time, x, solution


if __name__ == "__main__":
    main(500, 4, 0.9)
