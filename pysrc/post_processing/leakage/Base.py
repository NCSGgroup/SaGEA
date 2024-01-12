from abc import ABC, abstractmethod

from pysrc.auxiliary.core.GRID import GRID
from pysrc.post_processing.filter.Base import SHCFilter
from pysrc.post_processing.harmonic.Harmonic import Harmonic


class Leakage(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def apply_to(self, grids):
        pass

    @abstractmethod
    def format(self):
        pass


def filter_grids(grid, shc_filter: SHCFilter, harmonic: Harmonic):
    shc = harmonic.analysis(GRID(grid, lat=harmonic.lat, lon=harmonic.lon))

    shc_filtered = shc_filter.apply_to(shc)
    grid_filtered = harmonic.synthesis(shc_filtered)

    return grid_filtered
