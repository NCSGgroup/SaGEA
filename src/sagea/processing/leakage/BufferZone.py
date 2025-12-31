import numpy as np

from sagea.processing.filter.GetSHCFilter import get_filter
from sagea.processing.filter.Base import SHCFilter
from sagea.processing.Harmonic import Harmonic
from sagea.utils import MathTool


class BufferZoneConfig:
    def __init__(self):

        self.basin_map = None
        self.filter = None
        self.harmonic = None
        self.basin_acreage = None

        self.threshold = 0.1
        self.buffer_type = "shrink"  # "shrink", "expand"

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

    def set_threshold(self, threshold):
        assert 0 <= threshold <= 1
        self.threshold = threshold
        return self

    def set_buffer_type(self, buffer_type: str):
        assert buffer_type in ("shrink", "expand")
        self.buffer_type = buffer_type
        return self


class BufferZone():
    def __init__(self):
        self.configuration = BufferZoneConfig()

    def apply_to(self, gqij):
        buffered_basin_map = self.__get_buffered_basin()

        f_predicted = MathTool.global_integral(gqij * buffered_basin_map) / MathTool.get_acreage(buffered_basin_map)

        return f_predicted

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

        cqlm_bar, sqlm_bar = self.configuration.harmonic.analysis(basin_bar)

        cqlm_bar_f, sqlm_bar_f = self.configuration.filter.apply_to(cqlm_bar, sqlm_bar)
        basin_bar_filtered = self.configuration.harmonic.synthesis(cqlm_bar_f, sqlm_bar_f)

        threshold = self.configuration.threshold
        if self.configuration.buffer_type == "shrink":
            basin_bar_filtered[np.where(basin_bar_filtered > threshold)] = 1
            basin_bar_filtered[np.where(basin_bar_filtered <= threshold)] = 0
        elif self.configuration.buffer_type == "expand":
            basin_bar_filtered[np.where(basin_bar_filtered > threshold)] = 0
            basin_bar_filtered[np.where(basin_bar_filtered <= threshold)] = 1
        else:
            assert False

        return 1 - basin_bar_filtered


def buffer_zone(grid_value, lat, lon, basin_mask, filter_method, filter_param, lmax_calc,
                buffer_type="shrink", threshold=0.1):
    """
    extract signal 'more inside' basin
    """

    '''prepare'''
    har = Harmonic(lmax=lmax_calc, lat=lat, lon=lon, grid_type=None)
    shc_filter = get_filter(method=filter_method, params=filter_param, lmax=lmax_calc)
    lk = BufferZone()

    lk.configuration.set_basin(basin_mask)
    lk.configuration.set_filter(shc_filter)
    lk.configuration.set_harmonic(har)
    lk.configuration.set_threshold(threshold)
    lk.configuration.set_buffer_type(buffer_type)

    '''run buffer_zone'''
    f_predicted = lk.apply_to(grid_value)
    return f_predicted
