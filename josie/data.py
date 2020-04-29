import numpy as np

from dataclasses import dataclass
from typing import Dict

StateData = Dict[str, np.ndarray]


@dataclass
class StateElement:
    """ An handy class to store the state of a simulation
    """

    time: float
    data: StateData
