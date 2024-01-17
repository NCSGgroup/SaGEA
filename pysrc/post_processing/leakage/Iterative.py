import pathlib

import numpy as np

from pysrc.auxiliary.core.GRID import GRID
from pysrc.auxiliary.core.SHC import SHC
from pysrc.auxiliary.load_file.LoadL2SH import load_SH_simple
from pysrc.post_processing.leakage.Base import Leakage, filter_grids
from pysrc.post_processing.filter.Base import SHCFilter
from pysrc.post_processing.harmonic.Harmonic import Harmonic
from pysrc.auxiliary.tools.MathTool import MathTool


class IterativeConfig:
    def __init__(self):
        self.basin_map = None
        self.basin_acreage = None
        self.filter = None
        self.prefilter = None
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
            self.basin_map = basin

        self.basin_acreage = MathTool.get_acreage(self.basin_map)

        return self

    def set_filter(self, shc_filter: SHCFilter):
        self.filter = shc_filter

        return self.filter

    def set_prefilter(self, shc_filter: SHCFilter):
        self.prefilter = shc_filter

        return self.filter

    def set_shc_unfiltered(self, shc: SHC):
        self.shc_unfiltered = shc

        return self


class Iterative(Leakage):
    def __init__(self):
        super().__init__()
        self.configuration = IterativeConfig()

    def apply_to(self, grids: GRID):
        f_filtered = MathTool.global_integral(grids.data * self.configuration.basin_map)

        leakage_c = self.__get_leakage()

        return f_filtered - leakage_c

    def __get_leakage(self):
        if self.configuration.prefilter is None:
            self.configuration.prefilter = self.configuration.filter

        shc_prefiltered = self.configuration.prefilter.apply_to(self.configuration.shc_unfiltered)

        grids_prefiltered = self.configuration.harmonic.synthesis(shc_prefiltered)
        grids_prefiltered_outside = grids_prefiltered.data * (1 - self.configuration.basin_map)

        shc_prefiltered = self.configuration.harmonic.analysis(
            GRID(grids_prefiltered_outside, self.configuration.harmonic.lat, self.configuration.harmonic.lon)
        )

        shc_iter_filtered = self.configuration.filter.apply_to(shc_prefiltered)
        grids_iter_filtered = self.configuration.harmonic.synthesis(shc_iter_filtered)

        leakage_c = MathTool.global_integral(grids_iter_filtered.data * self.configuration.basin_map)

        return leakage_c

    def format(self):
        return 'Iterative'
