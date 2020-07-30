import numpy as np

from dataclasses import dataclass
from .state import State


@dataclass
class Neighbour:
    """A class representing the set of neighbours to some values.
    It ships the values of the fields in the neighbour cells, together with the
    face normals and the face surfaces"""

    values: State
    normals: np.ndarray
    surfaces: np.ndarray
