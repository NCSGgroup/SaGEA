#!/use/bin/env python
# coding=utf-8
# @Author  : Shuhao Liu
# @Time    : 2026/7/23 13:25 
# @File    : _TCH.py

from __future__ import annotations

import itertools
from enum import Enum

import numpy as np
from scipy.optimize import minimize

from sagea import GRD, SHC


class TCHMode(Enum):
    """
    TCH solution mode.

    KKT:
        Estimate the variance-covariance matrix using a constrained
        optimization formulation based on the Karush-Kuhn-Tucker condition.

    OLS:
        Estimate error variances from pairwise difference variances using
        ordinary least squares.
    """
    KKT = "KKT"
    OLS = "OLS"


class TCHNegativeVariancePolicy(Enum):
    """
    Strategy for handling negative variance estimates.

    NAN:
        Negative variance estimates are returned as NaN.
        This is recommended for scientific diagnostics.

    CLIP:
        Negative variance estimates are clipped to zero.

    ABS:
        Absolute values are used.
        This keeps behavior close to the original implementation, but it may
        hide violations of the TCH assumptions.
    """
    NAN = "NAN"
    CLIP = "CLIP"
    ABS = "ABS"


def _parse_enum(value, enum_class, name: str):
    """
    Parse Enum from either an Enum member or a string.
    """
    if isinstance(value, enum_class):
        return value

    if isinstance(value, str):
        key = value.upper()
        for member in enum_class:
            if key == member.name.upper() or key == member.value.upper():
                return member

    valid = [member.value for member in enum_class]
    raise ValueError(f"Invalid {name}: {value}. Valid options are {valid}.")


class TCHConfig:
    def __init__(self):
        self.__mode = TCHMode.OLS
        self.__negative_variance_policy = TCHNegativeVariancePolicy.NAN
        self.__min_valid_obs = 3
        self.__raise_on_optimization_failure = False

    def set_mode(self, mode: TCHMode | str):
        """
        Set TCH solution mode.

        Parameters
        ----------
        mode : TCHMode | str
            TCHMode.KKT or TCHMode.OLS.
        """
        self.__mode = _parse_enum(mode, TCHMode, "TCH mode")
        return self

    def get_mode(self):
        return self.__mode

    def set_negative_variance_policy(
            self,
            policy: TCHNegativeVariancePolicy | str
    ):
        """
        Set strategy for handling negative variance estimates.
        """
        self.__negative_variance_policy = _parse_enum(
            policy,
            TCHNegativeVariancePolicy,
            "negative variance policy"
        )
        return self

    def get_negative_variance_policy(self):
        return self.__negative_variance_policy

    def set_min_valid_obs(self, min_valid_obs: int):
        """
        Set the minimum number of valid samples required for TCH estimation.
        """
        if not isinstance(min_valid_obs, int):
            raise TypeError("min_valid_obs must be an integer.")

        if min_valid_obs < 3:
            raise ValueError("min_valid_obs must be greater than or equal to 3.")

        self.__min_valid_obs = min_valid_obs
        return self

    def get_min_valid_obs(self):
        return self.__min_valid_obs

    def set_raise_on_optimization_failure(self, flag: bool):
        """
        Whether to raise RuntimeError when KKT optimization fails.

        If False, NaN variances will be returned when optimization fails.
        """
        self.__raise_on_optimization_failure = bool(flag)
        return self

    def get_raise_on_optimization_failure(self):
        return self.__raise_on_optimization_failure


class TCH:
    def __init__(self):
        self.configuration = TCHConfig()

        self.datasets = None
        self.xn = None  # number of datasets/products/institutes
        self.xm = None  # length of each dataset/time series

    def set_datasets(self, *x):
        """
        Set input datasets.

        Parameters
        ----------
        *x : array-like
            At least three 1-D arrays with the same length.

        Notes
        -----
        For the KKT mode, the last dataset is used as the reference dataset,
        following the convention in the original implementation.
        """
        if len(x) < 3:
            raise ValueError("At least three datasets are required for TCH.")

        lengths = [len(xx) for xx in x]
        if len(set(lengths)) != 1:
            raise ValueError(f"All datasets must have the same length, got {lengths}.")

        self.xn = len(x)
        self.xm = lengths[0]

        if self.xm < self.configuration.get_min_valid_obs():
            raise ValueError(
                f"The length of each dataset is {self.xm}, "
                f"but min_valid_obs is {self.configuration.get_min_valid_obs()}."
            )

        self.datasets = np.asarray(x, dtype=float)

        if self.datasets.ndim != 2:
            raise ValueError(
                "TCH.set_datasets expects 1-D datasets. "
                f"Got stacked shape {self.datasets.shape}."
            )

        return self

    def __get_valid_datasets(self):
        """
        Remove time samples containing NaN or Inf in any dataset.

        Returns
        -------
        data : ndarray, shape (n_datasets, n_valid_samples)
        """
        if self.datasets is None:
            raise RuntimeError("Datasets have not been set.")

        data = self.datasets.copy()

        valid_mask = np.all(np.isfinite(data), axis=0)
        data = data[:, valid_mask]

        min_valid_obs = self.configuration.get_min_valid_obs()

        if data.shape[1] < min_valid_obs:
            return None

        return data

    @staticmethod
    def __apply_negative_variance_policy(
            var_array,
            policy: TCHNegativeVariancePolicy
    ):
        """
        Handle negative variance estimates.
        """
        var_array = np.asarray(var_array, dtype=float).copy()

        finite_mask = np.isfinite(var_array)
        if np.any(finite_mask):
            finite_abs_max = np.nanmax(np.abs(var_array[finite_mask]))
            tol = 1e-12 * max(finite_abs_max, 1.0)

            tiny_negative = (var_array < 0) & (var_array > -tol)
            var_array[tiny_negative] = 0.0

        if policy == TCHNegativeVariancePolicy.NAN:
            var_array[var_array < 0] = np.nan
        elif policy == TCHNegativeVariancePolicy.CLIP:
            var_array = np.maximum(var_array, 0.0)
        elif policy == TCHNegativeVariancePolicy.ABS:
            var_array = np.abs(var_array)
        else:
            raise ValueError(f"Unsupported negative variance policy: {policy}")

        return var_array

    def __run_with_OLS(self):
        """
        TCH estimation using ordinary least squares.

        Model
        -----
        For N products, the pairwise difference variance satisfies:

            Var(x_i - x_j) = sigma_i^2 + sigma_j^2

        if the errors of different products are mutually uncorrelated.

        The system is solved as:

            A @ sigma = b

        where sigma contains the unknown error variances.
        """
        data = self.__get_valid_datasets()
        if data is None:
            return np.full(self.xn, np.nan)

        nset, nobs = data.shape

        combination_of_index = list(itertools.combinations(range(nset), 2))
        n_combination = len(combination_of_index)

        A_mat = np.zeros((n_combination, nset), dtype=float)
        var_diff_array = np.zeros(n_combination, dtype=float)

        for row, (i, j) in enumerate(combination_of_index):
            A_mat[row, i] = 1.0
            A_mat[row, j] = 1.0

            diff = data[i] - data[j]

            # Use sample variance, consistent with covariance estimation.
            var_diff_array[row] = np.var(diff, ddof=1)

        # More stable than inv(A.T @ A) @ A.T @ b.
        var_array, *_ = np.linalg.lstsq(A_mat, var_diff_array, rcond=None)

        policy = self.configuration.get_negative_variance_policy()
        var_array = self.__apply_negative_variance_policy(var_array, policy)

        return var_array

    def __run_with_KKT2(self):
        """
        TCH estimation using KKT-based constrained optimization.

        Notes
        -----
        The last dataset is used as the reference dataset. This follows the
        structure of your original implementation.

        This implementation avoids in-place modification of self.datasets and
        checks the optimizer status.
        """
        data = self.__get_valid_datasets()
        if data is None:
            return np.full(self.xn, np.nan)

        nset, nobs = data.shape

        # Shape: nobs * nset.
        x_mat = data.T.copy()

        # De-mean each product.
        x_mat = x_mat - np.mean(x_mat, axis=0, keepdims=True)

        # Difference relative to the last product.
        # Shape: nobs * (nset - 1).
        y_mat = x_mat[:, :-1] - x_mat[:, -1][:, None]
        y_mat = y_mat - np.mean(y_mat, axis=0, keepdims=True)

        # Sample covariance matrix of difference series.
        s_mat = y_mat.T @ y_mat / (nobs - 1)

        u = np.ones(nset - 1, dtype=float)
        s_mat_inv = np.linalg.pinv(s_mat)

        def _build_r_hat(x):
            """
            Build the upper-left block of the full covariance matrix.

            x[:-1] : covariance between the first N-1 products and the reference.
            x[-1]  : variance of the reference product.
            """
            r = x[:-1]
            rnn = x[-1]

            return (
                    s_mat
                    - rnn * np.outer(u, u)
                    + np.outer(u, r)
                    + np.outer(r, u)
            )

        def objective(x):
            """
            Objective function for KKT-based estimation.
            """
            r_hat = _build_r_hat(x)
            r_hat_lower = np.tril(r_hat, k=-1)

            # Penalize off-diagonal elements of the upper-left block and
            # reference covariances.
            return np.sum(r_hat_lower ** 2) + np.sum(x[:-1] ** 2)

        def constraint_ineq(x):
            """
            Inequality constraint for SLSQP.

            scipy.optimize.minimize requires constraint fun(x) >= 0.
            """
            r = x[:-1]
            rnn = x[-1]
            v = r - rnn * u

            return v.T @ s_mat_inv @ v - rnn

        # Initial condition.
        denom = 2.0 * (u.T @ s_mat_inv @ u)

        if not np.isfinite(denom) or denom <= 0:
            rnn0 = np.nanmean(np.diag(s_mat))
            if not np.isfinite(rnn0) or rnn0 <= 0:
                rnn0 = 1.0
        else:
            rnn0 = 1.0 / denom

        r_ini = np.concatenate([
            np.zeros(nset - 1, dtype=float),
            np.array([rnn0], dtype=float)
        ])

        constraints = ({
                           "type": "ineq",
                           "fun": constraint_ineq
                       },)

        solution = minimize(
            objective,
            r_ini,
            method="SLSQP",
            constraints=constraints,
            options={
                "maxiter": 1000,
                "ftol": 1e-12,
                "disp": False
            }
        )

        if not solution.success:
            msg = f"KKT optimization failed: {solution.message}"

            if self.configuration.get_raise_on_optimization_failure():
                raise RuntimeError(msg)

            return np.full(nset, np.nan)

        r_with_rnn = solution.x
        r = r_with_rnn[:-1]
        rnn = r_with_rnn[-1]

        r_mat = np.zeros((nset, nset), dtype=float)

        r_mat[:-1, :-1] = (
                s_mat
                - rnn * np.outer(u, u)
                + np.outer(u, r)
                + np.outer(r, u)
        )

        r_mat[-1, :-1] = r
        r_mat[:-1, -1] = r
        r_mat[-1, -1] = rnn

        var_array = np.diag(r_mat)

        policy = self.configuration.get_negative_variance_policy()
        var_array = self.__apply_negative_variance_policy(var_array, policy)

        return var_array

    def __run_with_KKT(self):
        """
        TCH estimation using the KKT-based constrained optimization.

        This version preserves the numerical form of the original implementation
        as much as possible, including the determinant-based scaling factor K.
        """
        data = self.__get_valid_datasets()
        if data is None:
            return np.full(self.xn, np.nan)

        nset, nobs = data.shape

        # Shape: nobs * nset.
        # Use copy to avoid modifying self.datasets.
        x_mat = data.T.copy()

        # De-average.
        x_mat = x_mat - np.mean(x_mat, axis=0, keepdims=True)

        # The last product is used as the reference.
        y_mat = x_mat[:, :-1] - x_mat[:, -1][:, None]

        # De-average difference series.
        y_mat = y_mat - np.mean(y_mat, axis=0, keepdims=True)

        # Sample covariance matrix of difference series.
        s_mat = y_mat.T @ y_mat / (nobs - 1)

        u = np.ones((nset - 1,), dtype=float)
        s_mat_inv = np.linalg.pinv(s_mat)

        det_s = np.linalg.det(s_mat)

        # Preserve original K definition as much as possible.
        # However, guard against exact zero or non-finite determinant.
        if not np.isfinite(det_s) or det_s == 0:
            if self.configuration.get_raise_on_optimization_failure():
                raise RuntimeError(
                    f"Invalid determinant of s_mat in KKT mode: det={det_s}"
                )
            return np.full(nset, np.nan)

        K = det_s ** (1 - nset)

        if not np.isfinite(K) or K == 0:
            if self.configuration.get_raise_on_optimization_failure():
                raise RuntimeError(f"Invalid K in KKT mode: K={K}")
            return np.full(nset, np.nan)

        def objective(x):
            """
            Same numerical form as the original implementation.
            """
            r_hat = (
                    s_mat
                    - x[-1] * np.outer(u, u)
                    + np.outer(u, x[:-1])
                    + np.outer(x[:-1], u)
            )

            r_hat_tril = np.tril(r_hat, k=-1)

            return (np.sum(r_hat_tril ** 2) + np.sum(x[:-1] ** 2)) / (K ** 2)

        def constraint_ineq1(x):
            """
            Same numerical form as the original implementation.

            SLSQP uses constraint fun(x) >= 0.
            """
            v = x[:-1] - x[-1] * u

            return -1.0 / K * (
                    x[-1] - v.T @ s_mat_inv @ v
            )

        # Initial condition: same as original implementation.
        denom = 2.0 * u.T @ s_mat_inv @ u

        if not np.isfinite(denom) or denom == 0:
            if self.configuration.get_raise_on_optimization_failure():
                raise RuntimeError(f"Invalid initial denominator: {denom}")
            return np.full(nset, np.nan)

        r_ini = np.concatenate([
            np.zeros((nset - 1,), dtype=float),
            np.array([denom ** (-1)], dtype=float)
        ])

        con1 = {
            "type": "ineq",
            "fun": constraint_ineq1
        }

        # To reproduce the old behavior more closely, do not set strict options.
        solution = minimize(
            objective,
            r_ini,
            method="SLSQP",
            constraints=con1
        )

        if not solution.success:
            if self.configuration.get_raise_on_optimization_failure():
                raise RuntimeError(f"KKT optimization failed: {solution.message}")

            # If you want to exactly mimic the old implementation,
            # comment out the next line and continue using solution.x.
            return np.full(nset, np.nan)

        r_with_rnn = solution.x
        r = r_with_rnn[:-1]
        rnn = r_with_rnn[-1]

        r_mat = np.zeros((nset, nset), dtype=float)

        r_mat[:-1, :-1] = (
                s_mat
                - rnn * np.outer(u, u)
                + np.outer(u, r)
                + np.outer(r, u)
        )

        r_mat[-1, :-1] = r
        r_mat[:-1, -1] = r
        r_mat[-1, -1] = rnn

        var_array = np.diag(r_mat)

        policy = self.configuration.get_negative_variance_policy()
        var_array = self.__apply_negative_variance_policy(var_array, policy)

        return var_array

    def get_variance(self):
        """
        Return TCH-based error variance estimates.
        """
        mode = self.configuration.get_mode()

        if mode == TCHMode.KKT:
            var_array = self.__run_with_KKT()
        elif mode == TCHMode.OLS:
            var_array = self.__run_with_OLS()
        else:
            raise ValueError(f"Unsupported TCH mode: {mode}")

        return var_array

    def get_std(self):
        """
        Return TCH-based error standard deviation estimates.
        """
        var_array = self.get_variance()
        return np.sqrt(var_array)


def _validate_array_inputs(*dataset):
    """
    Validate numpy array inputs.
    """
    if len(dataset) < 3:
        raise ValueError("At least three datasets are required for TCH.")

    shapes = [np.asarray(d).shape for d in dataset]

    if len(set(shapes)) != 1:
        raise ValueError(f"All input arrays must have the same shape, got {shapes}.")

    if len(shapes[0]) < 1:
        raise ValueError("Input arrays must have at least one dimension.")

    return shapes[0]


def _tch_for_array(
        *dataset,
        mode: TCHMode,
        negative_variance_policy: TCHNegativeVariancePolicy,
        min_valid_obs: int,
        raise_on_optimization_failure: bool
):
    """
    TCH for numpy arrays.

    Input convention
    ----------------
    Each dataset should have shape:

        (time,)

    or

        (time, dim1, dim2, ...)

    The first axis is treated as the sample/time dimension.

    Returns
    -------
    If the input arrays are 1-D:
        ndarray with shape (n_datasets,)

    If the input arrays are multi-dimensional:
        tuple of ndarray, one for each dataset.
    """
    mode = _parse_enum(mode, TCHMode, "TCH mode")
    negative_variance_policy = _parse_enum(
        negative_variance_policy,
        TCHNegativeVariancePolicy,
        "negative variance policy"
    )

    input_shape = _validate_array_inputs(*dataset)

    data = np.asarray(dataset, dtype=float)

    nset = data.shape[0]
    ntime = data.shape[1]

    if ntime < min_valid_obs:
        raise ValueError(
            f"The length of the first dimension is {ntime}, "
            f"but min_valid_obs is {min_valid_obs}."
        )

    tch_obj = TCH()
    tch_obj.configuration.set_mode(mode)
    tch_obj.configuration.set_negative_variance_policy(negative_variance_policy)
    tch_obj.configuration.set_min_valid_obs(min_valid_obs)
    tch_obj.configuration.set_raise_on_optimization_failure(
        raise_on_optimization_failure
    )

    # 1-D time series case.
    if len(input_shape) == 1:
        tch_obj.set_datasets(*[data[i] for i in range(nset)])
        return tch_obj.get_std()

    # Multi-dimensional case.
    spatial_shape = input_shape[1:]
    spatial_size = int(np.prod(spatial_shape))

    data_2d = data.reshape(nset, ntime, spatial_size)

    tch_results_2d = np.full((nset, spatial_size), np.nan, dtype=float)

    for i in range(spatial_size):
        tch_obj.set_datasets(*[data_2d[j, :, i] for j in range(nset)])
        tch_results_2d[:, i] = tch_obj.get_std()

    tch_results = tch_results_2d.reshape((nset, *spatial_shape))

    return tuple(tch_results[i] for i in range(nset))


def _tch_for_grd(
        *grid: GRD,
        mode: TCHMode,
        negative_variance_policy: TCHNegativeVariancePolicy,
        min_valid_obs: int,
        raise_on_optimization_failure: bool
):
    """
    TCH for GRD objects.
    """
    nlength = len(grid)

    dataset = [grid[i].value for i in range(nlength)]

    tch_results = _tch_for_array(
        *dataset,
        mode=mode,
        negative_variance_policy=negative_variance_policy,
        min_valid_obs=min_valid_obs,
        raise_on_optimization_failure=raise_on_optimization_failure
    )

    if isinstance(tch_results, tuple):
        return tuple(
            GRD(tch_results[i], lat=grid[0].lat, lon=grid[0].lon)
            for i in range(nlength)
        )

    # For the rare case where GRD.value is 1-D and _tch_for_array returns
    # ndarray with shape (n_products,).
    return tuple(
        GRD(tch_results[i], lat=grid[0].lat, lon=grid[0].lon)
        for i in range(nlength)
    )


def _tch_for_shc(
        *shc: SHC,
        mode: TCHMode,
        negative_variance_policy: TCHNegativeVariancePolicy,
        min_valid_obs: int,
        raise_on_optimization_failure: bool
):
    """
    TCH for SHC objects.
    """
    nlength = len(shc)

    dataset = [shc[i].value for i in range(nlength)]

    tch_results = _tch_for_array(
        *dataset,
        mode=mode,
        negative_variance_policy=negative_variance_policy,
        min_valid_obs=min_valid_obs,
        raise_on_optimization_failure=raise_on_optimization_failure
    )

    if isinstance(tch_results, tuple):
        return tuple(
            SHC(tch_results[i])
            for i in range(nlength)
        )

    # For the rare case where SHC.value is 1-D and _tch_for_array returns
    # ndarray with shape (n_products,).
    return tuple(
        SHC(tch_results[i])
        for i in range(nlength)
    )


def tch(
        *dataset: np.ndarray | GRD | SHC,
        mode: TCHMode | str = TCHMode.OLS,
        negative_variance_policy: TCHNegativeVariancePolicy | str =
        TCHNegativeVariancePolicy.NAN,
        min_valid_obs: int = 3,
        raise_on_optimization_failure: bool = False
):
    """
    TCH-based error standard deviation estimation.

    Parameters
    ----------
    *dataset : np.ndarray | GRD | SHC
        At least three datasets.

        For numpy arrays, each dataset should have shape:

            (time,)

        or

            (time, dim1, dim2, ...)

        The first axis is interpreted as the sample/time dimension.

    mode : TCHMode | str, default TCHMode.OLS
        TCH solution mode.

        TCHMode.OLS:
            Ordinary least squares based on pairwise difference variances.

        TCHMode.KKT:
            KKT-based constrained optimization.
            The last dataset is used as the reference dataset.

    negative_variance_policy : TCHNegativeVariancePolicy | str, default NAN
        Strategy for handling negative variance estimates.

        "NAN":
            Return NaN for negative variance estimates.
            Recommended for scientific diagnostics.

        "CLIP":
            Clip negative variance estimates to zero.

        "ABS":
            Use absolute values.
            This is close to the previous implementation but may hide
            violations of the TCH assumptions.

    min_valid_obs : int, default 3
        Minimum number of valid samples after removing NaN/Inf.

    raise_on_optimization_failure : bool, default False
        Only used in KKT mode.
        If True, raise RuntimeError when the optimizer fails.
        If False, return NaN estimates.

    Returns
    -------
    If input is 1-D numpy arrays:
        ndarray with shape (n_datasets,)

    If input is multi-dimensional numpy arrays:
        tuple of ndarray, one for each dataset.

    If input is GRD:
        tuple of GRD.

    If input is SHC:
        tuple of SHC.

    Notes
    -----
    The basic TCH model assumes:

        x_i = t + e_i

    and

        Cov(e_i, e_j) = 0, i != j

    Under this assumption:

        Var(x_i - x_j) = Var(e_i) + Var(e_j)

    The returned values are empirical random error standard deviations.
    They should not be interpreted as total uncertainty if systematic bias,
    leakage error, filtering error, or inter-product correlated errors exist.
    """
    if len(dataset) < 3:
        raise ValueError("At least three datasets are required for TCH.")

    mode = _parse_enum(mode, TCHMode, "TCH mode")
    negative_variance_policy = _parse_enum(
        negative_variance_policy,
        TCHNegativeVariancePolicy,
        "negative variance policy"
    )

    if not isinstance(min_valid_obs, int):
        raise TypeError("min_valid_obs must be an integer.")

    if min_valid_obs < 3:
        raise ValueError("min_valid_obs must be greater than or equal to 3.")

    first = dataset[0]

    if isinstance(first, np.ndarray):
        if not all(isinstance(d, np.ndarray) for d in dataset):
            raise TypeError("All input datasets must be numpy arrays.")

        return _tch_for_array(
            *dataset,
            mode=mode,
            negative_variance_policy=negative_variance_policy,
            min_valid_obs=min_valid_obs,
            raise_on_optimization_failure=raise_on_optimization_failure
        )

    elif isinstance(first, GRD):
        if not all(isinstance(d, GRD) for d in dataset):
            raise TypeError("All input datasets must be GRD objects.")

        return _tch_for_grd(
            *dataset,
            mode=mode,
            negative_variance_policy=negative_variance_policy,
            min_valid_obs=min_valid_obs,
            raise_on_optimization_failure=raise_on_optimization_failure
        )

    elif isinstance(first, SHC):
        if not all(isinstance(d, SHC) for d in dataset):
            raise TypeError("All input datasets must be SHC objects.")

        return _tch_for_shc(
            *dataset,
            mode=mode,
            negative_variance_policy=negative_variance_policy,
            min_valid_obs=min_valid_obs,
            raise_on_optimization_failure=raise_on_optimization_failure
        )

    else:
        raise TypeError(
            "Unsupported dataset type. "
            "Expected np.ndarray, GRD, or SHC, "
            f"but got {type(first)}."
        )
