from sagea.processing.leakage.Additive import Additive
from sagea.processing.leakage.BufferZone import BufferZone
from sagea.processing.leakage.DataDriven import DataDriven
from sagea.processing.leakage.ForwardModeling import ForwardModeling
from sagea.processing.leakage.Iterative import Iterative
from sagea.processing.leakage.Multiplicative import Multiplicative
from sagea.processing.leakage.Scaling import Scaling
from sagea.processing.leakage.ScalingGrid import ScalingGrid

from sagea.constant import LeakageMethod


def get_leakage_corrector(method: LeakageMethod):
    """
    :param method: LeakageType
    """

    if method == LeakageMethod.Multiplicative:
        leakage = Multiplicative()

    elif method == LeakageMethod.Additive:
        leakage = Additive()

    elif method == LeakageMethod.Scaling:
        leakage = Scaling()

    elif method == LeakageMethod.ScalingGrid:
        leakage = ScalingGrid()

    elif method == LeakageMethod.Iterative:
        leakage = Iterative()

    elif method == LeakageMethod.DataDriven:
        leakage = DataDriven()

    elif method == LeakageMethod.ForwardModeling:
        leakage = ForwardModeling()

    elif method == LeakageMethod.BufferShrink:
        leakage = BufferZone()

    elif method == LeakageMethod.BufferExpand:
        leakage = BufferZone()

    else:
        assert False

    return leakage
