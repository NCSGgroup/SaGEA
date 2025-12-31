import numpy as np

from sagea.processing.Harmonic import Harmonic
from sagea.processing.filter.GetSHCFilter import get_filter
from sagea.processing.leakage.Base import filter_grids
from sagea.utils import MathTool


# class Additive(ModelDriven):
#     def apply_to(self, gqij, leakage=True, bias=True, get_grid=True):
#         basin_map = self.configuration.basin_map
#         f_filtered = MathTool.global_integral(gqij * basin_map) / MathTool.get_acreage(basin_map)
#
#         f_predicted = copy.deepcopy(f_filtered)
#
#         if leakage:
#             leakage_c_m = self._get_leakage()
#             f_predicted -= leakage_c_m
#
#         if bias:
#             bias_c_m = self._get_bias()
#             f_predicted += bias_c_m
#
#         if get_grid:
#             return f_predicted[:, None, None] * self.configuration.basin_map
#         else:
#             return f_predicted
#
#     def format(self):
#         return 'addictive'


def additive(grid_value, lat, lon, basin_mask, reference, filter_method, filter_param, lmax_calc):
    """
    f_predicted = f_filtered - leakage + bias.
        bias: estimated from reference,
        leakage: estimated from reference.
    """

    '''prepare'''
    basin_mask = np.array([basin_mask])
    har = Harmonic(lmax=lmax_calc, lat=lat, lon=lon, grid_type=None)
    shc_filter = get_filter(method=filter_method, params=filter_param, lmax=lmax_calc)

    '''get f_filtered'''
    f_filtered = MathTool.global_integral(grid_value * basin_mask) / MathTool.get_acreage(basin_mask)

    '''get bias'''
    basin_filtered = filter_grids(basin_mask, shc_filter, harmonic=har)[0]
    bias = MathTool.global_integral(reference * (basin_mask - basin_filtered)) / MathTool.get_acreage(basin_mask)

    '''get leakage'''
    basin_outside = 1 - basin_mask
    model_outside_basin = reference * basin_outside
    model_outside_filtered = filter_grids(model_outside_basin, shc_filter, harmonic=har)
    leakage = MathTool.global_integral(model_outside_filtered * basin_mask) / MathTool.get_acreage(basin_mask)

    '''get f_predicted'''
    f_predicted = f_filtered - leakage + bias

    return f_predicted
