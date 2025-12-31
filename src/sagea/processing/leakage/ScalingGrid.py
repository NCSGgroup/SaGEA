import numpy as np

from sagea.processing.Harmonic import Harmonic
from sagea.processing.filter.GetSHCFilter import get_filter
from sagea.processing.leakage.Base import filter_grids
from sagea.utils import MathTool


def __scale_function(x, a):
    return a * x


def scaling_grid(grid_value, lat, lon, basin_mask, reference, filter_method, filter_param, lmax_calc):
    """
    f_predicted = k * f_filtered (for each gridded cell).
        k: estimated from reference.
    """

    '''prepare'''
    basin_mask = np.array([basin_mask])
    har = Harmonic(lmax=lmax_calc, lat=lat, lon=lon, grid_type=None)
    shc_filter = get_filter(method=filter_method, params=filter_param, lmax=lmax_calc)

    '''get gridded k'''
    model_filtered = filter_grids(reference, shc_filter, harmonic=har)
    model_shape = np.shape(reference)[1:]

    model_1d = np.array([reference[i].flatten() for i in range(len(reference))])
    model_filtered_1d = np.array([model_filtered[i].flatten() for i in range(len(model_filtered))])

    scale_1d = np.ones_like(model_1d[0])
    for i in range(model_1d.shape[1]):
        z = MathTool.curve_fit(__scale_function, model_filtered_1d[:, i], model_1d[:, i])
        scale_1d[i] = z[0][0, 0]

    scale_gridded = scale_1d.reshape(model_shape)

    '''get f_predicted'''
    f_predicted_gridded = scale_gridded * grid_value
    basin_size = MathTool.get_acreage(basin_mask)
    f_predicted = MathTool.global_integral(f_predicted_gridded * basin_mask) / basin_size

    return f_predicted
