import h5py
import matplotlib.pyplot as plt
import numpy as np

from pysrc.auxiliary.aux_tool.FileTool import FileTool
from pysrc.auxiliary.aux_tool.MathTool import MathTool

from pysrc.post_processing.harmonic.Harmonic import Harmonic, CoreSHC, CoreGRID


class SHC(CoreSHC):
    def __init__(self, c, s=None):
        super().__init__(c, s)

    def to_grid(self, grid_space=None):
        if grid_space is None:
            grid_space = int(180 / self.get_lmax())

        lat, lon = MathTool.get_global_lat_lon_range(grid_space)

        lmax = self.get_lmax()
        har = Harmonic(lat, lon, lmax, option=1)

        cqlm, sqlm = self.get_cs2d()
        grid_data = har.synthesis_for_csqlm(cqlm, sqlm)
        grid = GRID(grid_data, lat, lon, option=1)

        return grid


class GRID(CoreGRID):
    def __init__(self, grid, lat, lon, option=0):
        super().__init__(grid, lat, lon, option)

    def to_SHC(self, lmax=None):
        grid_space = self.get_grid_space()

        if lmax is None:
            lmax = int(180 / grid_space)

        lat, lon = MathTool.get_global_lat_lon_range(grid_space)

        har = Harmonic(lat, lon, lmax, option=1)

        grid_data = self.data
        cqlm, sqlm = har.analysis_for_gqij(grid_data)
        shc = SHC(cqlm, sqlm)

        return shc


if __name__ == '__main__':
    pass
