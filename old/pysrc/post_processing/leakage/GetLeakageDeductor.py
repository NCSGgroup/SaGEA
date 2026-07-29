from pysrc.auxiliary.preference.EnumClasses import LeakageMethod
from pysrc.post_processing.leakage.Addictive import Addictive
from pysrc.post_processing.leakage.BufferZone import BufferZone
from pysrc.post_processing.leakage.DataDriven import DataDriven
from pysrc.post_processing.leakage.ForwardModeling import ForwardModeling
from pysrc.post_processing.leakage.Iterative import Iterative
from pysrc.post_processing.leakage.Multiplicative import Multiplicative
from pysrc.post_processing.leakage.Scaling import Scaling
from pysrc.post_processing.leakage.ScalingGrid import ScalingGrid


def get_leakage_deductor(method: LeakageMethod):
    """
    :param method: LeakageType
    """

    if method == LeakageMethod.Multiplicative:
        leakage = Multiplicative()

    elif method == LeakageMethod.Additive:
        leakage = Addictive()

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

    elif method == LeakageMethod.BufferZone:
        leakage = BufferZone()

    else:
        assert False, 'leakage false'

    return leakage
