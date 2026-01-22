#!/use/bin/env python
# coding=utf-8
# @Author  : Shuhao Liu
# @Time    : 2026/1/2 10:52 
# @File    : least_square.py
import numpy as np
import warnings
from typing import Union, List, Tuple, Callable


class GeneralLS:
    """
    A Generic Linear Least Squares (LLS) Solver supporting multiple RHS.
    Solves Y = AX + E for matrices Y (N x K) and X (M x K).
    """

    def __init__(self, t: np.ndarray, y: Union[np.ndarray, List], sigma: np.ndarray = None):
        """
        :param t: Independent variable (e.g., time), shape (N,)
        :param y: Dependent variables.
                  Shape (N,) for single series,
                  Shape (N, K) for K simultaneous series.
        :param sigma: Measurement errors for W matrix construction. shape (N,).
                      Assumption: All K columns of Y share the same weights W.
        """
        self.t = np.asarray(t)

        # Ensure y is at least 2D internally (N, K) even if input is (N,)
        _y = np.asarray(y)
        if _y.ndim == 1:
            self.y = _y[:, np.newaxis]  # (N, 1)
        else:
            self.y = _y  # (N, K)

        self.N, self.K = self.y.shape

        if len(self.t) != self.N:
            raise ValueError(f"Shape mismatch: t has {len(self.t)}, y has {self.N} rows.")

        # Design matrix container
        self._basis_funcs = []
        self._col_names = []

        # Weight Matrix Construction (Shared across all Y columns)
        if sigma is not None:
            sigma = np.asarray(sigma)
            if sigma.shape[0] != self.N:
                raise ValueError("Sigma length must match observations N.")
            # Inverse variance weighting
            self.W = np.diag(1.0 / (sigma ** 2))
            self.weighted = True
        else:
            self.W = None
            self.weighted = False

    def add_basis(self, name: str, func_callable: Callable[[np.ndarray], np.ndarray]):
        """
        Adds a generic basis function column to the design matrix.
        :param name: String identifier (e.g., 'Trend', 'Bias').
        :param func_callable: Function f(t) -> array (N,).
        """
        self._col_names.append(name)
        col_data = func_callable(self.t)

        # Determine shape safety
        col_data = np.asarray(col_data)
        if col_data.shape[0] != self.N:
            raise ValueError(f"Basis function {name} produced length {col_data.shape[0]}, expected {self.N}.")

        self._basis_funcs.append(col_data)

    def solve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Solves the system.

        Returns:
            params: (M, K) matrix. Row = parameter, Col = specific y series.
            covariance: (M, M) matrix (Valid for the structural model).
            residuals: (N, K) matrix.
            names: List of parameter names.
        """
        # 1. Build Design Matrix A (N x M)
        if not self._basis_funcs:
            raise ValueError("No basis functions added.")
        A = np.column_stack(self._basis_funcs)
        M = A.shape[1]

        # 2. Check Degree of Freedom Stability
        if self.N < M:
            raise ValueError(f"Under-determined system: N ({self.N}) < Parameters ({M}).")

        # Optional Warning for statistical stability
        if self.N < M * 5:
            warnings.warn(f"[Instability Risk] Low redundancy: N={self.N}, Params={M}.")

        # 3. Construct Normal Equations (A^T W A) x = (A^T W) Y
        # Matrix Shapes:
        # A: (N, M)
        # W: (N, N)
        # Y: (N, K)
        # LHS: (M, M)
        # RHS: (M, K) <--- This handles multiple Y columns at once

        if self.weighted:
            # P = A.T @ W (Precompute for reuse)
            P = A.T @ self.W  # (M, N)
            LHS = P @ A  # (M, M)
            RHS = P @ self.y  # (M, K)
        else:
            LHS = A.T @ A  # (M, M)
            RHS = A.T @ self.y  # (M, K)

        # 4. Numerical Solution
        try:
            # (M, M) * (M, K) -> (M, K)
            # Use 'solve' instead of 'inv' for better numerical stability
            params = np.linalg.solve(LHS, RHS)

            # Covariance Matrix of Parameters (M, M)
            # Cov(x) = (A^T W A)^-1
            # Note: This is the unscaled covariance structure.
            # For strict statistics, one usually scales this by MSE for each column K.
            cov_structure = np.linalg.inv(LHS)

        except np.linalg.LinAlgError:
            raise ValueError("Singular matrix. Basis functions are likely collinear.")

        # 5. Residuals (N, K)
        model_prediction = A @ params
        residuals = self.y - model_prediction

        return params, cov_structure, residuals, self._col_names


class PeriodicLS(GeneralLS):
    """
    Subclass tailored for Time Series / Harmonic Analysis.
    Inherits the Multi-Y support.
    """

    def __init__(self, t, y, sigma=None):
        super().__init__(t, y, sigma)
        # Registry for spectral health checks
        self.sinusoid_components = []

    def add_bias(self):
        self.add_basis("Bias", lambda t: np.ones_like(t))

    def add_linear(self):
        self.add_basis("Linear_trend", lambda t: t)

    def add_period(self, period: float, name: str = None):
        """
        Adds cos/sin pair. Metadata stored for validation.
        """
        if period <= 1e-9:
            raise ValueError("Period must be > 0.")

        freq = 1.0 / period
        if name is None:
            name = f"T{period:.2f}"

        # 1. Metadata Registration
        self.sinusoid_components.append({
            "name": name,
            "freq": freq,
            "period": period
        })

        # 2. Add Basis (Math)
        self.add_basis(f"{name}_c", lambda t: np.cos(2 * np.pi * freq * t))
        self.add_basis(f"{name}_s", lambda t: np.sin(2 * np.pi * freq * t))

    def check_spectral_health(self) -> List[str]:
        """
        Checks Nyquist (Aliasing) and Rayleigh (Resolution) criteria.
        Returns a list of warning strings.
        """
        warnings_found = []

        # Analyze Time Vector
        # Use np.ptp (peak-to-peak) for range
        t_span = np.ptp(self.t)
        if t_span == 0: return ["Time span is zero."]

        # Calculate sampling intervals
        t_sorted = np.sort(self.t)
        dt = np.diff(t_sorted)
        dt_median = np.median(dt)

        # --- Rayleigh Criterion Check ---
        # Minimal frequency separation detectable
        f_rayleigh = 1.0 / t_span

        comps = self.sinusoid_components
        for i in range(len(comps)):
            for j in range(i + 1, len(comps)):
                c1 = comps[i]
                c2 = comps[j]

                diff_freq = abs(c1['freq'] - c2['freq'])

                # Using a slightly strict factor (e.g., 0.9) to be safe
                if diff_freq < f_rayleigh:
                    warnings_found.append(
                        f"[Rayleigh Conflict] '{c1['name']}' ({c1['period']:.1f}) & '{c2['name']}' ({c2['period']:.1f}).\n"
                        f"  Frequency Diff: {diff_freq:.2e} < Limit: {f_rayleigh:.2e} (1/T_obs).\n"
                        f"  Condition number will likely explode."
                    )

        # --- Nyquist Criterion Check ---
        # Highest observable freq is 1 / (2 * dt)
        f_nyquist = 1.0 / (2.0 * dt_median)

        for c in comps:
            if c['freq'] > f_nyquist:
                warnings_found.append(
                    f"[Nyquist Conflict] '{c['name']}' period {c['period']:.2f} is too short.\n"
                    f"  Freq: {c['freq']:.2f} > Limit: {f_nyquist:.2f} (1/2dt).\n"
                    f"  This signal will be aliased."
                )

        return warnings_found


# ==========================================
# Demonstration: Multiple Y Support
# ==========================================
if __name__ == "__main__":

    # 1. Setup Time and Ground Truth
    t_days = np.linspace(2005, 2005 + 365 * 10, 240)  # 2 Years of data

    # Series 1 (K=0): North Component - Has strong Annual signal
    y1 = 5.0 * np.cos(2 * np.pi * t_days / 365.25) + np.random.randn(240)

    # Series 2 (K=1): East Component - Has strong Semiannual signal (182.6 days)
    y2 = 3.0 * np.sin(2 * np.pi * t_days / 182.625) + np.random.randn(240)

    y_grid = np.zeros((len(t_days), 180, 360))
    trend_map = np.random.random(size=(180, 360))
    annual_amp_map = np.random.random(size=(180, 360))
    annual_pha_map = np.random.random(size=(180, 360))
    semiannual_amp_map = np.random.random(size=(180, 360))
    semiannual_pha_map = np.random.random(size=(180, 360))
    s2_amp_map = np.random.random(size=(180, 360))
    s2_pha_map = np.random.random(size=(180, 360))

    y_grid = t_days[:, None, None] * trend_map + annual_amp_map * np.sin(
        2 * np.pi * t_days[:, None, None] / 365.25 + annual_pha_map) + semiannual_amp_map * np.sin(
        2 * np.pi * t_days[:, None, None] / 182.625 + semiannual_pha_map) + s2_amp_map * np.sin(
        2 * np.pi * t_days[:, None, None] / 161 + s2_pha_map)

    Y_matrix = y_grid.reshape(len(t_days), -1)

    # Combine into (N, 2) matrix
    # Y_matrix = np.column_stack((y1, y2))

    print(f"Input Shape Y: {Y_matrix.shape}")  # Expect (200, 2)

    # 2. Init Solver
    model = PeriodicLS(t_days, Y_matrix)

    # 3. Build Model (Add components common to both series)
    #    Even if 'Annual' isn't strong in East, LS will just calculate a small coeff.
    model.add_bias()
    model.add_linear()
    model.add_period(period=365.25, name="Annual")  # 1 year
    model.add_period(period=182.625, name="SemiAnnual")  # 6 months
    model.add_period(period=161, name="S2_aliasing")

    # 4. Check Physics
    msgs = model.check_spectral_health()
    if msgs:
        for m in msgs: print(m)
    else:
        print("Spectral Physics OK.")

    # 5. Solve Both simultaneously
    # params shape will be (M, 2)
    coeffs, cov, resid, names = model.solve()

    print(coeffs[0].shape)

    bias = coeffs[0]
    linear = coeffs[1]
    annual_c, annual_s = coeffs[2], coeffs[3]
    semiannual_c, semiannual_s = coeffs[4], coeffs[5]

    print(names)
    trend_map_pred = coeffs[1].reshape(180, 360)
    annual_amp_pred = np.sqrt(coeffs[2] ** 2 + coeffs[3] ** 2).reshape(180, 360)
    semiannual_amp_pred = np.sqrt(coeffs[4] ** 2 + coeffs[5] ** 2).reshape(180, 360)
    s2_amp_pred = np.sqrt(coeffs[6] ** 2 + coeffs[7] ** 2).reshape(180, 360)

    pass
