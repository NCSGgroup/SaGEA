import pathlib

import numpy as np

from pysrc.auxiliary.core.GRID import GRID
from pysrc.auxiliary.core.SHC import SHC
from pysrc.auxiliary.load_file.LoadL2SH import load_SH_simple
from pysrc.post_processing.leakage.Base import Leakage, filter_grids
from pysrc.post_processing.filter.Base import SHCFilter
from pysrc.post_processing.harmonic.Harmonic import Harmonic
from pysrc.auxiliary.tools.MathTool import MathTool


class DataDrivenConfig:
    def __init__(self):
        self.basin_map = None
        self.basin_acreage = None
        self.filter = None
        self.harmonic = None
        self.shc_unfiltered = None

    def set_harmonic(self, har: Harmonic):
        self.harmonic = har
        return self

    def set_basin(self, basin: SHC or pathlib.WindowsPath):
        assert self.harmonic is not None, "set harmonic before setting basin."

        har = self.harmonic

        if type(basin) is pathlib.WindowsPath:
            lmax = self.harmonic.lmax
            basin_clm, basin_slm = load_SH_simple(basin, key='', lmax=lmax, lmcs_in_queue=(1, 2, 3, 4))
            self.basin_map = har.synthesis(SHC(basin_clm, basin_slm)).data[0]

        else:
            self.basin_map = har.synthesis(basin).data[0]

        self.basin_acreage = MathTool.get_acreage(self.basin_map)

        return self

    def set_filter(self, shc_filter: SHCFilter):
        self.filter = shc_filter

        return self.filter

    def set_shc_unfiltered(self, shc: SHC):
        self.shc_unfiltered = shc

        return self


class DataDriven(Leakage):
    def __init__(self):
        super().__init__()
        self.configuration = DataDrivenConfig()

    # def config(self, *, basin, cqlm_unf, sqlm_unf, shc_filter: SHCFilter, harmonic: Harmonic):
    #     self.basin = basin
    #     self.cqlm_unf = cqlm_unf
    #     self.sqlm_unf = sqlm_unf
    #     self.filter = shc_filter
    #     self.har = harmonic
    #
    #     return self

    def apply_to(self, grids: GRID):
        f_filtered = MathTool.global_integral(grids.data * self.configuration.basin_map)

        leakage_c = self.__get_leakage()
        deviation = self.__get_deviation()

        return f_filtered - deviation - leakage_c

    def __get_leakage(self):
        grids_unf = self.configuration.harmonic.synthesis(self.configuration.shc_unfiltered)
        grids_outside_unf = grids_unf.data * (1 - self.configuration.basin_map)

        shc_outside_unfiltered = self.configuration.harmonic.analysis(
            GRID(grids_outside_unf, self.configuration.harmonic.lat, self.configuration.harmonic.lon)
        )
        shc_outside_filtered = self.configuration.filter.apply_to(shc_outside_unfiltered)

        grids_outside_filtered = self.configuration.harmonic.synthesis(shc_outside_filtered)

        leakage_c = MathTool.global_integral(grids_outside_filtered.data * self.configuration.basin_map)

        return leakage_c

    def __get_deviation(self):
        grids_unf = self.configuration.harmonic.synthesis(self.configuration.shc_unfiltered)

        basin_acreage = MathTool.get_acreage(self.configuration.basin_map)

        basin_average = MathTool.global_integral(grids_unf.data * self.configuration.basin_map) / basin_acreage

        deviation_field = grids_unf.data * self.configuration.basin_map - np.einsum('ijk,i->ijk',
                                                                                    np.ones_like(grids_unf.data),
                                                                                    basin_average)

        grids_dev_f = filter_grids(deviation_field, self.configuration.filter, self.configuration.harmonic)

        deviation_filtered = MathTool.global_integral(grids_dev_f.data * self.configuration.basin_map)

        return deviation_filtered

    def format(self):
        return 'Data-driven'