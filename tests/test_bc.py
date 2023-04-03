# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from josie.bc import _ConstantDirichlet, Dirichlet, _ConstantNeumann, Neumann


def test_constant_dirichlet():
    bc = Dirichlet(0)

    assert isinstance(bc, _ConstantDirichlet)


def test_constant_callable(mocker):
    def callable(cells, t):
        return cells.values + 1

    bc = Dirichlet(callable, constant=True)
    assert isinstance(bc, _ConstantDirichlet)

    cells = mocker.Mock()
    values = mocker.PropertyMock(return_value=np.ones((4, 4)))

    type(cells).values = values

    bc.init(cells)

    assert np.array_equal(bc._value, (cells.values + 1))


def test_constant_neumann():
    bc = Neumann(0)

    assert isinstance(bc, _ConstantNeumann)
