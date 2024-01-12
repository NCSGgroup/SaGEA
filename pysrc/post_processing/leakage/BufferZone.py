import pathlib

import matplotlib.pyplot as plt
import numpy as np

from pysrc.auxiliary.core.GRID import GRID
from pysrc.auxiliary.core.SHC import SHC
from pysrc.auxiliary.load_file.LoadL2SH import load_SH_simple
from pysrc.auxiliary.tools.MathTool import MathTool
from pysrc.post_processing.filter.Base import SHCFilter
from pysrc.post_processing.harmonic.Harmonic import Harmonic
from pysrc.post_processing.leakage.Base import Leakage


class BufferZoneConfig:
    def __init__(self):

        self.basin_map = None
        self.filter = None
        self.harmonic = None
        self.basin_acreage = None

    def set_harmonic(self, har: Harmonic):
        self.harmonic = har
        return self

    def set_basin(self, basin: SHC or pathlib.WindowsPath or np.ndarray):
        assert self.harmonic is not None, "set harmonic before setting basin."


        har = self.harmonic

        if type(basin) is pathlib.WindowsPath:
            lmax = self.harmonic.lmax
            basin_clm, basin_slm = load_SH_simple(basin, key='', lmax=lmax, lmcs_in_queue=(1, 2, 3, 4))
            self.basin_map = har.synthesis(SHC(basin_clm, basin_slm)).data[0]

        elif type(basin) is np.ndarray:
            assert basin.ndim == 2
            self.basin_map = basin

        else:
            self.basin_map = har.synthesis(basin).data[0]

        self.basin_acreage = MathTool.get_acreage(self.basin_map)

        return self

    def set_filter(self, shc_filter: SHCFilter):
        self.filter = shc_filter

        return self.filter


class BufferZone(Leakage):
    def __init__(self):
        super().__init__()
        self.configuration = BufferZoneConfig()

        self.buffered_basin_map = None

    def apply_to(self, grids):
        self.buffered_basin_map = self.__get_buffered_basin()

        f_filtered_in_buffer_zone = MathTool.global_integral(grids.data * self.buffered_basin_map)
        return f_filtered_in_buffer_zone

    def __get_buffered_basin(self):
        basin_bar = 1 - self.configuration.basin_map

        basin_bar[np.where(basin_bar > 0.5)] = 1
        basin_bar[np.where(basin_bar <= 0.5)] = 0

        lat, lon = self.configuration.harmonic.lat, self.configuration.harmonic.lon
        shc_bar = self.configuration.harmonic.analysis(GRID(basin_bar, lat, lon))

        shc_bar_filtered = self.configuration.filter.apply_to(shc_bar)
        basin_bar_filtered = self.configuration.harmonic.synthesis(shc_bar_filtered).data[0]

        threshold = 0.1
        basin_bar_filtered[np.where(basin_bar_filtered > threshold)] = 1
        basin_bar_filtered[np.where(basin_bar_filtered <= threshold)] = 0

        return 1 - basin_bar_filtered

    def format(self):
        return "buffer zone"


if __name__ == '__main__':
    from pysrc.auxiliary.tools.FileTool import FileTool
    from pysrc.auxiliary.scripts.PlotGrids import plot_grids
    from pysrc.post_processing.filter.Gaussian import Gaussian

    bfz = BufferZone()

    grid_space = 1
    lmax = 60
    lat, lon = MathTool.get_global_lat_lon_range(grid_space)
    har = Harmonic(lat, lon, lmax, option=1)

    shc_filter = Gaussian()
    radius = 500
    shc_filter.configuration.set_lmax(lmax)
    shc_filter.configuration.set_filtering_radius(radius)

    bfz.configuration.set_harmonic(har)
    bfz.configuration.set_filter(shc_filter)
    bfz.configuration.set_basin(FileTool.get_project_dir('data/auxiliary/ocean360_grndline.sh'))

    ocean = bfz.apply_to(0).buffered_basin_map

    plot_grids(
        ocean,
        lat, lon, 0, 1
    )