import numpy as np

from sagea.utils import MathTool
from sagea.processing.filter.Base import SHCFilter
from sagea.processing.Harmonic import Harmonic
from sagea.processing.leakage.Base import Leakage


class BufferZoneConfig:
    def __init__(self):

        self.basin_map = None
        self.filter = None
        self.harmonic = None
        self.basin_acreage = None

    def set_harmonic(self, harmonic: Harmonic):
        self.harmonic = harmonic
        return self

    def set_basin(self, basin: np.ndarray):
        assert type(basin) in (np.ndarray,)

        if type(basin) is np.ndarray:
            assert basin.ndim == 2
            self.basin_map = basin

        else:
            assert False

        self.basin_acreage = MathTool.get_acreage(self.basin_map)

        return self

    def set_filter(self, shc_filter: SHCFilter):
        self.filter = shc_filter

        return self.filter


class BufferZone(Leakage):
    def __init__(self):
        super().__init__()
        self.configuration = BufferZoneConfig()

    def apply_to(self, gqij, get_grid=True):
        buffered_basin_map = self.__get_buffered_basin()

        f_predicted = MathTool.global_integral(gqij * buffered_basin_map) / MathTool.get_acreage(buffered_basin_map)

        if get_grid:
            return f_predicted[:, None, None] * self.configuration.basin_map
        else:
            return f_predicted

        # return f_filtered_in_buffer_zone

    def get_buffer(self):
        # buffered_basin_map = self.__get_buffered_basin()
        # colat_rad, lon_rad = self.configuration.harmonic.lat, self.configuration.harmonic.lon
        # lat, lon = MathTool.get_lat_lon_degree(colat_rad, lon_rad)
        #
        # buffered_basin_grid = GRID(buffered_basin_map, lat, lon)
        #
        # return buffered_basin_grid
        return self.__get_buffered_basin()

    def __get_buffered_basin(self):
        basin_bar = 1 - self.configuration.basin_map

        basin_bar[np.where(basin_bar > 0.5)] = 1
        basin_bar[np.where(basin_bar <= 0.5)] = 0

        lat, lon = self.configuration.harmonic.lat, self.configuration.harmonic.lon
        # shc_bar = self.configuration.harmonic.analysis(GRID(basin_bar, lat, lon))
        cqlm_bar, sqlm_bar = self.configuration.harmonic.analysis(basin_bar)

        cqlm_bar_f, sqlm_bar_f = self.configuration.filter.apply_to(cqlm_bar, sqlm_bar)
        basin_bar_filtered = self.configuration.harmonic.synthesis(cqlm_bar_f, sqlm_bar_f)

        threshold = 0.1
        basin_bar_filtered[np.where(basin_bar_filtered > threshold)] = 1
        basin_bar_filtered[np.where(basin_bar_filtered <= threshold)] = 0

        return 1 - basin_bar_filtered

    def format(self):
        return "buffer zone"
