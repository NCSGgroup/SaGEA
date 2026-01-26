import matplotlib.pyplot as plt

from sagea.utils import TimeTool
from sagea.sgmath.least_square import PeriodicLS

import numpy as np
from typing import List, Dict, Optional
from datetime import date


def grid_tide_de_aliasing(
        grid_data: np.ndarray,
        dates: List[date],
        tide_periods: Optional[Dict[str, float]] = None
) -> np.ndarray:
    """
    Remove specific tidal alias signals from 3D grid data using the periodicLS class.

    Parameters
    ----------
    grid_data : np.ndarray
        3D Array with shape (Time, Lat, Lon).
    dates : list of date
        List of datetime.date objects corresponding to the Time dimension.
    tide_periods : dict, optional
        Dictionary of tides to remove.
        Key: Tide name, Value: Period in DAYS.
        e.g., {'S2': 161.0}.
        If None, returns original data.

    Returns
    -------
    np.ndarray
        De-aliased grid data with shape (Time, Lat, Lon).
    """

    year_frac = np.array(TimeTool.convert_date_format(
        dates, input_type=TimeTool.DateFormat.ClassDate, output_type=TimeTool.DateFormat.YearFraction
    ))

    one_grid_shape = grid_data.shape[1:]
    Y_matrix = grid_data.reshape(len(year_frac), -1)

    model = PeriodicLS(year_frac, Y_matrix)

    model.add_bias()
    model.add_linear()
    model.add_period(period=1., name="Annual")
    model.add_period(period=0.5, name="SemiAnnual")
    for key in tide_periods.keys():
        model.add_period(period=tide_periods[key] / 365.25, name=key)

    msgs = model.check_spectral_health()
    if msgs:
        for m in msgs: print(m)
    else:
        pass

    coeffs, _, resid, names = model.solve()

    bias = coeffs[0]
    linear = coeffs[1]
    annual_c, annual_s = coeffs[2], coeffs[3]
    semiannual_c, semiannual_s = coeffs[4], coeffs[5]
    s2_c, s2_s = coeffs[6], coeffs[7]

    yf_expand = year_frac[:, None]
    Y_matrix_dealiasing = bias + yf_expand * linear + annual_c * np.cos(2 * np.pi * yf_expand) + annual_s * np.sin(
        2 * np.pi * yf_expand) + semiannual_c * np.cos(4 * np.pi * yf_expand) + semiannual_s * np.sin(
        4 * np.pi * yf_expand)

    grid_data_dealiasing = Y_matrix_dealiasing.reshape((len(year_frac), *one_grid_shape)) + resid.reshape(
        (len(year_frac), *one_grid_shape))

    return grid_data_dealiasing


if __name__ == '__main__':
    trend = np.array(
        [[1.5, 1.6, 1.9],
         [1.4, 1.55, 1.92]]
    )

    annual_amp = np.array(
        [[5.1, 5.3, 4.4],
         [4.1, 4.6, 4.9]]
    )

    annual_pha = np.array(
        [[1.1, 1.2, 1.3],
         [1.5, 1.1, 0.4]]
    )

    semiannual_amp = np.array(
        [[2.1, 2.3, 1.4],
         [1.1, 1.6, 1.9]]
    )

    semiannual_pha = np.array(
        [[1.0, 1.1, 1.2],
         [1.3, 1.5, 0.9]]
    )

    s2_amp = np.array(
        [[0.1, 0.3, 0.4],
         [0.1, 0.6, 0.9]]
    )

    s2_pha = np.array(
        [[1.9, 1.3, 1.2],
         [1.7, 1.4, 0.6]]
    )

    year_frac = np.linspace(2005, 2015, 120)

    dates = TimeTool.convert_date_format(year_frac, input_type=TimeTool.DateFormat.YearFraction,
                                         output_type=TimeTool.DateFormat.ClassDate)
    yf_exp = year_frac[:, None, None]

    grid_data = yf_exp * trend + annual_amp * np.sin(2 * np.pi * yf_exp + annual_pha) + semiannual_amp * np.sin(
        4 * np.pi * yf_exp + semiannual_pha) + s2_amp * np.sin(2 * np.pi * yf_exp / (161 / 365.25) + s2_pha)
    grid_data -= np.mean(grid_data, axis=0)

    grid_data += np.random.normal(size=grid_data.shape, scale=1, loc=0)

    grid_data_de_s2 = grid_tide_de_aliasing(grid_data, dates, tide_periods=dict(S2=161))

    grid_s2 = grid_data - grid_data_de_s2

    print(grid_data.shape)

    plt.plot(year_frac, grid_data[:, 0, 0])
    plt.plot(year_frac, grid_data_de_s2[:, 0, 0])
    plt.plot(year_frac, grid_s2[:, 0, 0])
    plt.show()
