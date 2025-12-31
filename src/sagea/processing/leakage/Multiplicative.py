import numpy as np

from sagea.processing.Harmonic import Harmonic
from sagea.processing.filter.GetSHCFilter import get_filter
from sagea.processing.leakage.Base import filter_grids
from sagea.utils import MathTool


# class Multiplicative(ModelDriven):
#     def apply_to(self, grids, get_grid=False, bias=True, leakage=True):
#         """
#         :param grids:
#         :param get_grid:
#         :param bias: if False, it will not scale the bias
#         :param leakage: if False, it will not reduce the leakage_c_m
#         """
#
#         basin_map = self.configuration.basin_map
#
#         f_filtered = MathTool.global_integral(grids * basin_map) / MathTool.get_acreage(basin_map)
#
#         f_predicted = copy.deepcopy(f_filtered)
#         if leakage:
#             leakage_c_m = self._get_leakage() / MathTool.get_acreage(basin_map)
#             f_predicted -= leakage_c_m
#
#         if bias:
#             scale = self._get_multiplicative_scale()
#             f_predicted *= scale
#
#         if get_grid:
#             return f_predicted[:, None, None] * self.configuration.basin_map
#         else:
#             return f_predicted
#
#     def get_scale(self):
#         return self._get_multiplicative_scale()
#
#     def format(self):
#         return 'multiplicative'


def multiplicative(grid_value, lat, lon, basin_mask, reference, filter_method, filter_param, lmax_calc):
    """
    f_predicted = k * (f_filtered - leakage).
        k: estimated from basin_mask,
        leakage: estimated from reference.
    """

    '''prepare'''
    basin_mask = np.array([basin_mask])
    har = Harmonic(lmax=lmax_calc, lat=lat, lon=lon, grid_type=None)
    shc_filter = get_filter(method=filter_method, params=filter_param, lmax=lmax_calc)

    '''get f_filtered'''
    f_filtered = MathTool.global_integral(grid_value * basin_mask) / MathTool.get_acreage(basin_mask)

    '''get k'''
    basin_mask_filtered = filter_grids(basin_mask, shc_filter, harmonic=har)[0]
    integral_basin_mask = MathTool.global_integral(basin_mask)
    integral_basin_mask_filtered = MathTool.global_integral(basin_mask_filtered * basin_mask)
    k = integral_basin_mask / integral_basin_mask_filtered

    '''get leakage'''
    basin_outside = 1 - basin_mask
    model_outside_basin = reference * basin_outside
    model_outside_filtered = filter_grids(model_outside_basin, shc_filter, harmonic=har)
    leakage = MathTool.global_integral(model_outside_filtered * basin_mask) / MathTool.get_acreage(basin_mask)

    '''get f_predicted'''
    f_predicted = k * (f_filtered - leakage)

    return f_predicted
