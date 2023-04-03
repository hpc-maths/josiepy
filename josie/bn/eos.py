# SPDX-FileCopyrightText: 2020-2023 JosiePy Development Team
#
# SPDX-License-Identifier: BSD-3-Clause

from josie.euler.eos import EOS as SinglePhaseEOS
from josie.twofluid.state import PhasePair


class TwoPhaseEOS(PhasePair):
    """An Abstract Base Class representing en EOS for a twophase system.  In
    particular two :class:`.euler.eos.EOS` instances for each phase need to be
    provided.

    You can access the EOS for a specified phase using the
    :meth:`__getitem__`

    """

    def __init__(self, phase1: SinglePhaseEOS, phase2: SinglePhaseEOS):
        """
        Parameters
        ----------
        phase1
            An instance of :class:`.euler.eos.EOS` representing the EOS for the
            single phase #1
        phase2
            An instance of :class:`.euler.eos.EOS` representing the EOS for the
            single phase #2
        """

        super().__init__(phase1, phase2)
