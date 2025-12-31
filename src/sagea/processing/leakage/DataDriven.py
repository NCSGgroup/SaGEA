import numpy as np

from sagea.processing.filter.GetSHCFilter import get_filter
from sagea.processing.leakage.Base import filter_grids
from sagea.processing.filter.Base import SHCFilter
from sagea.processing.Harmonic import Harmonic
from sagea.utils import MathTool


class DataDrivenConfig:
    def __init__(self):
        self.basin_map = None
        self.basin_acreage = None
        self.filter = None
        self.harmonic = None

        self.cqlm_unfiltered = None
        self.sqlm_unfiltered = None

    def set_harmonic(self, har: Harmonic):
        self.harmonic = har
        return self

    def set_basin(self, basin: np.ndarray):
        # assert self.harmonic is not None, "set harmonic before setting basin."

        types = (np.ndarray,)
        assert type(basin) in types, "basin should be a numpy array."

        self.basin_map = basin

        self.basin_acreage = MathTool.get_acreage(self.basin_map)

        return self

    def set_filter(self, shc_filter: SHCFilter):
        self.filter = shc_filter

        return self.filter

    def set_cs_unfiltered(self, cqlm, sqlm):
        self.cqlm_unfiltered = cqlm
        self.sqlm_unfiltered = sqlm

        return self


class DataDriven():
    def __init__(self):
        self.configuration = DataDrivenConfig()

    def apply_to(self, gqij, get_grid=False):
        basin_map = self.configuration.basin_map
        f_filtered = MathTool.global_integral(gqij * basin_map) / MathTool.get_acreage(basin_map)

        leakage_c = self.__get_leakage() / MathTool.get_acreage(basin_map)
        deviation = self.__get_deviation() / MathTool.get_acreage(basin_map)

        f_predicted = f_filtered - deviation - leakage_c

        if get_grid:
            return f_predicted[:, None, None] * self.configuration.basin_map
        else:
            return f_predicted

    def __get_leakage(self):
        har = self.configuration.harmonic
        cqlm_unf, sqlm_unf = self.configuration.cqlm_unfiltered, self.configuration.sqlm_unfiltered
        gqij_unf = har.synthesis(cqlm_unf, sqlm_unf)
        cs_filter = self.configuration.filter
        basin = self.configuration.basin_map

        gqij_outside_unf = gqij_unf * (1 - basin)

        cqlm_outside_unf, sqlm_outside_unf = har.analysis(gqij_outside_unf)

        cqlm_outside_f, sqlm_outside_f = cs_filter.apply_to(cqlm_outside_unf, sqlm_outside_unf)

        gqij_outside_f = har.synthesis(cqlm_outside_f, sqlm_outside_f)

        leakage_c = MathTool.global_integral(gqij_outside_f * basin)

        return leakage_c

    def __get_deviation(self):
        har = self.configuration.harmonic
        cqlm_unf, sqlm_unf = self.configuration.cqlm_unfiltered, self.configuration.sqlm_unfiltered
        gqij_unf = har.synthesis(cqlm_unf, sqlm_unf)
        basin = self.configuration.basin_map
        cs_filter = self.configuration.filter

        basin_acreage = self.configuration.basin_acreage

        basin_average = MathTool.global_integral(gqij_unf * self.configuration.basin_map) / basin_acreage

        deviation_field = gqij_unf * basin - np.einsum('ijk,i->ijk', np.ones_like(gqij_unf), basin_average)

        grids_dev_f = filter_grids(deviation_field, cs_filter, har)

        deviation_filtered = MathTool.global_integral(grids_dev_f * basin)

        return deviation_filtered


def data_driven(grid_value, lat, lon, basin_mask, shc_unfiltered, filter_method, filter_param, lmax_calc):
    """

    """

    '''prepare'''
    basin_mask = np.array([basin_mask])
    har = Harmonic(lmax=lmax_calc, lat=lat, lon=lon, grid_type=None)
    shc_filter = get_filter(method=filter_method, params=filter_param, lmax=lmax_calc)

    lk = DataDriven()
    lk.configuration.set_cs_unfiltered(*shc_unfiltered.cs2d)

    lk.configuration.set_basin(basin_mask)
    lk.configuration.set_filter(shc_filter)
    lk.configuration.set_harmonic(har)

    '''run data-driven'''
    f_predicted = lk.apply_to(grid_value)

    return f_predicted
