import numpy as np

from pysrc.post_processing.leakage.Base import Leakage
from pysrc.post_processing.filter.Base import SHCFilter
from pysrc.post_processing.harmonic.Harmonic import Harmonic
from pysrc.auxiliary.aux_tool.MathTool import MathTool


class IterativeConfig:
    def __init__(self):
        self.basin_map = None
        self.basin_acreage = None
        self.filter = None
        self.prefilter = None
        self.harmonic = None
        self.cqlm_unfiltered = None
        self.sqlm_unfiltered = None

    def set_harmonic(self, har: Harmonic):
        self.harmonic = har
        return self

    def set_basin(self, basin: np.ndarray):
        # har = self.harmonic
        #
        # if type(basin) is pathlib.WindowsPath:
        #     lmax = self.harmonic.lmax
        #     basin_clm, basin_slm = load_SHC(basin, key='', lmax=lmax, lmcs_in_queue=(1, 2, 3, 4)).get_cs2d()
        #     self.basin_map = har.synthesis(SHC(basin_clm, basin_slm)).value[0]
        #     self.basin_map = har.synthesis(SHC(basin_clm, basin_slm)).value[0]
        #
        # else:
        #     self.basin_map = basin
        self.basin_map = basin

        self.basin_acreage = MathTool.get_acreage(self.basin_map)

        return self

    def set_filter(self, shc_filter: SHCFilter):
        self.filter = shc_filter

        return self.filter

    def set_prefilter(self, shc_filter: SHCFilter):
        self.prefilter = shc_filter

        return self.filter

    def set_cs_unfiltered(self, cqlm, sqlm):
        self.cqlm_unfiltered = cqlm
        self.sqlm_unfiltered = sqlm

        return self


class Iterative(Leakage):
    def __init__(self):
        super().__init__()
        self.configuration = IterativeConfig()

    def apply_to(self, gqij, get_grid=False):
        basin_map = self.configuration.basin_map
        f_filtered = MathTool.global_integral(gqij * basin_map) / MathTool.get_acreage(basin_map)

        leakage_c = self.__get_leakage() / MathTool.get_acreage(basin_map)

        f_predicted = f_filtered - leakage_c

        if get_grid:
            return f_predicted[:, None, None] * self.configuration.basin_map
        else:
            return f_predicted

    def __get_leakage(self):
        if self.configuration.prefilter is None:
            self.configuration.prefilter = self.configuration.filter

        basin_map = self.configuration.basin_map

        cqlm_unf, sqlm_unf = self.configuration.cqlm_unfiltered, self.configuration.sqlm_unfiltered
        cqlm_f, sqlm_f = self.configuration.prefilter.apply_to(cqlm_unf, sqlm_unf)

        gqij_prefiltered = self.configuration.harmonic.synthesis_for_csqlm(cqlm_f, sqlm_f)
        gqij_prefiltered_outside = gqij_prefiltered * (1 - basin_map)

        cqlm_pref, sqlm_pref = self.configuration.harmonic.analysis_for_gqij(gqij_prefiltered_outside)

        cqlm_iter_filtered, sqlm_iter_filtered = self.configuration.filter.apply_to(cqlm_pref, sqlm_pref)

        gqij_iter_filtered = self.configuration.harmonic.synthesis_for_csqlm(cqlm_iter_filtered, sqlm_iter_filtered)

        leakage_c = MathTool.global_integral(gqij_iter_filtered * basin_map)

        return leakage_c

    def format(self):
        return 'Iterative'
