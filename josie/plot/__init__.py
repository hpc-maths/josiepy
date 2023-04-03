# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

# from .mayavi import MayaviBackend  # noqa F401
from .matplotlib import MatplotlibBackend  # noqa F401

__all__ = ["MatplotlibBackend"]

# TODO: Make it configurable
DefaultBackend = MatplotlibBackend
