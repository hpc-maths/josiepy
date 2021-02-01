import numpy as np

from typing import Sequence, Tuple, Union

PointType = Union[np.ndarray, Sequence[float]]
MeshIndex = Union[Tuple[Union[int, slice, np.ndarray], ...], np.ndarray]
