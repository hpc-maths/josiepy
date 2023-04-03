# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from typing import Sequence, Tuple, Union

PointType = Union[np.ndarray, Sequence[float]]
MeshIndex = Union[Tuple[Union[int, slice, np.ndarray], ...], np.ndarray]
