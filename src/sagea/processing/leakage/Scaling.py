import numpy as np

from sagea.processing.Harmonic import Harmonic
from sagea.processing.filter.GetSHCFilter import get_filter
from sagea.processing.leakage.Base import filter_grids
from sagea.utils import MathTool


def __scale_function(x, a):
    return a * x


def scaling(grid_value, lat, lon, basin_mask, reference, filter_method, filter_param, lmax_calc):
    """
    f_predicted = k * f_filtered.
        k: estimated from reference.
    """

    '''prepare'''
    basin_mask = np.array([basin_mask])
    har = Harmonic(lmax=lmax_calc, lat=lat, lon=lon, grid_type=None)
    shc_filter = get_filter(method=filter_method, params=filter_param, lmax=lmax_calc)

    '''get f_filtered'''
    basin_size = MathTool.get_acreage(basin_mask)
    f_filtered = MathTool.global_integral(grid_value * basin_mask) / basin_size

    '''get k'''
    model_filtered = filter_grids(reference, shc_filter, harmonic=har)

    time_series_model = f_filtered
    time_series_model_filtered = MathTool.global_integral(model_filtered * basin_mask) / basin_size

    z = MathTool.curve_fit(__scale_function, time_series_model_filtered, time_series_model)

    k = z[0][0, 0]

    '''get f_predicted'''
    f_predicted = k * f_filtered

    return f_predicted
