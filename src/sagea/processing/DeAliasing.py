from sagea.utils import MathTool, TimeTool
from sagea.math.least_square import PeriodicLS

import numpy as np
from typing import List, Dict, Optional
from datetime import date


# def grid_tide_dealiasing(
#         grid_data: np.ndarray,
#         dates: List[date],
#         tide_periods: Optional[Dict[str, float]] = None
# ) -> np.ndarray:
#     """
#     Remove specific tidal alias signals from 3D grid data via least squares fitting.
#
#     Workflow:
#     1. Fit a model including: Trend + Annual + Semiannual + Specified Tides.
#     2. Isolate the tidal component from the fitted parameters.
#     3. Reconstruct the tidal signal and subtract it from the original data.
#
#     Parameters
#     ----------
#     grid_data : np.ndarray
#         3D Array with shape (Time, Lat, Lon).
#     dates : list of date
#         List of datetime.date objects corresponding to the Time dimension.
#     tide_periods : dict, optional
#         Dictionary of tides to remove.
#         Key: Tide name (str), Value: Period in DAYS (float).
#         e.g., {'S2': 161.0, 'K2': 1362.0}.
#         If None, the original data is returned without modification.
#
#     Returns
#     -------
#     np.ndarray
#         De-aliased grid data with shape (Time, Lat, Lon).
#         (Trend and annual/semiannual signals are PRESERVED).
#     """
#     # 0. Basic Validation
#     T, Lat_dim, Lon_dim = grid_data.shape
#
#     if len(dates) != T:
#         raise ValueError(f"Date length ({len(dates)}) must match Grid time dimension ({T}).")
#
#     if not tide_periods:
#         warnings.warn("No tide periods provided. Returning original data.")
#         return grid_data
#
#     # 1. Prepare Time Variables
#     # Convert dates to decimal years for the fitting model
#     # Assuming t_trans returns a list of floats
#     t_year = np.array(TimeTool.convert_date_format(
#         dates, TimeTool.DateFormat.ClassDate, TimeTool.DateFormat.YearFraction
#     ))
#
#     # Use relative time to improve numerical stability of the least squares fit
#     t_rel = t_year - t_year[0]
#
#     # 2. Construct Design Matrix (A)
#     # The model includes: Constant, Trend, Annual, Semiannual, and Custom Tides
#
#     # Define fixed base periods in YEARS: Annual (1.0), Semiannual (0.5)
#     base_periods = [1.0, 0.5]
#
#     # Convert input tide periods from DAYS to YEARS
#     # Formula: T_years = T_days / 365.25
#     tide_names = list(tide_periods.keys())
#     tide_T_years = [p / 365.25 for p in tide_periods.values()]
#
#     # Combine all periodic components for simultaneous fitting
#     all_periods = base_periods + tide_T_years
#
#     # 2.1 Static Terms: Constant and Linear Trend
#     const_col = np.ones((T, 1))
#     trend_col = t_rel.reshape(-1, 1)
#     A = np.concatenate((const_col, trend_col), axis=1)
#
#     # 2.2 Periodic Terms: Cosine and Sine pairs for each period
#     for p in all_periods:
#         omega = 2 * np.pi / p
#         c_col = np.cos(omega * t_rel).reshape(-1, 1)
#         s_col = np.sin(omega * t_rel).reshape(-1, 1)
#         A = np.concatenate((A, c_col, s_col), axis=1)
#
#     # 3. Flatten Spatial Dimensions for Vectorized Processing
#     # Transform grid from (Time, Lat, Lon) -> (Time, Spatial_Points)
#     # This allows using linalg.lstsq to solve for all pixels simultaneously
#     L_flat = grid_data.reshape(T, -1)
#
#     # Masking strategy: Only solve for pixels that have NO NaNs in the time series
#     # (Pixels with NaN are usually ocean/land masks or boundaries)
#     valid_mask = ~np.isnan(L_flat).any(axis=0)
#     L_compute = L_flat[:, valid_mask]
#
#     # If all pixels are masked/invalid, return early
#     if L_compute.shape[1] == 0:
#         return grid_data
#
#     # 4. Solve Weighted Least Squares (OLS here as weight=1)
#     # Solve A * X = L for X
#     # X_params shape: (N_parameters, N_valid_pixels)
#     X_params, _, _, _ = np.linalg.lstsq(A, L_compute, rcond=None)
#
#     # 5. Extract ONLY the Tide Components for reconstruction
#     # Structure of Matrix A columns:
#     # [0: Const, 1: Trend,
#     #  2,3: Annual(Cos,Sin), 4,5: Semiannual(Cos,Sin),
#     #  6,7: Tide1(Cos,Sin)... ]
#
#     # Count parameters we want to KEEP (Const + Trend + Base Periodic)
#     # These should NOT be subtracted
#     n_static = 2  # const + trend
#     n_base_cols = len(base_periods) * 2
#
#     # The starting column index for the Tide parameters
#     start_idx_tide = n_static + n_base_cols
#
#     # Slice the Design Matrix for tides
#     # Shape: (T, 2 * N_tides)
#     A_tide = A[:, start_idx_tide:]
#
#     # Slice the Estimated Coefficients for tides
#     # Shape: (2 * N_tides, N_valid_pixels)
#     X_tide = X_params[start_idx_tide:, :]
#
#     # 6. Reconstruct the Aliasing Signal
#     # Matrix multiplication: (T, Params) @ (Params, Pixels) -> (T, Pixels)
#     tide_signal_valid = A_tide @ X_tide
#
#     # 7. Subtract Tides from Original Data
#     # Create a full-sized correction matrix (default zeros)
#     correction_matrix = np.zeros_like(L_flat)
#
#     # Fill in the computed tide signals at valid pixels
#     correction_matrix[:, valid_mask] = tide_signal_valid
#
#     # Perform de-aliasing
#     dealiased_flat = L_flat - correction_matrix
#
#     # 8. Reshape back to 3D Grid dimensions
#     return dealiased_flat.reshape(T, Lat_dim, Lon_dim)
#
#
# import numpy as np
# import copy
# from typing import List, Dict, Optional
# from datetime import date


# Assuming periodicLS is imported from your local module

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

    pass
