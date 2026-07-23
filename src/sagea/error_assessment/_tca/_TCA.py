#!/use/bin/env python
# coding=utf-8
# @Author  : Shuhao Liu
# @Time    : 2026/7/23 13:54 
# @File    : _TCA.py

import itertools
from enum import Enum

import numpy as np
from scipy.optimize import least_squares

from sagea import GRD, SHC


class TCAMode(Enum):
    """
    CLASSIC:
        Classical three-product triple collocation.
        Exactly three datasets are required.

    ALL_TRIPLETS:
        For N >= 3 datasets, perform classical TCA for all possible triplets
        and aggregate the variance estimates by nanmedian.

    MTC:
        Multiple collocation based on fitting the off-diagonal covariance
        matrix with a rank-one signal covariance model.
    """
    CLASSIC = "CLASSIC"
    ALL_TRIPLETS = "ALL_TRIPLETS"
    NLS = "NLS"


class TCANegativeVariancePolicy(Enum):
    """
    How to handle negative variance estimates.

    NAN:
        Negative variance estimates are returned as NaN.
        Recommended for scientific diagnostics.

    CLIP:
        Negative variance estimates are clipped to zero.

    ABS:
        Absolute values are used.
        This is not recommended scientifically, but is provided for compatibility.
    """
    NAN = "NAN"
    CLIP = "CLIP"
    ABS = "ABS"


class TCAConfig:
    def __init__(self):
        self.__mode = TCAMode.ALL_TRIPLETS
        self.__negative_variance_policy = TCANegativeVariancePolicy.NAN
        self.__min_valid_obs = 3

    def set_mode(self, mode: TCAMode):
        if isinstance(mode, str):
            mode = TCAMode[mode.upper()]

        if mode not in TCAMode:
            raise ValueError(f"Invalid TCA mode: {mode}")

        self.__mode = mode
        return self

    def get_mode(self):
        return self.__mode

    def set_negative_variance_policy(self, policy: TCANegativeVariancePolicy):
        if isinstance(policy, str):
            policy = TCANegativeVariancePolicy[policy.upper()]

        if policy not in TCANegativeVariancePolicy:
            raise ValueError(f"Invalid negative variance policy: {policy}")

        self.__negative_variance_policy = policy
        return self

    def get_negative_variance_policy(self):
        return self.__negative_variance_policy

    def set_min_valid_obs(self, min_valid_obs: int):
        if min_valid_obs < 3:
            raise ValueError("min_valid_obs should be greater than or equal to 3.")

        self.__min_valid_obs = min_valid_obs
        return self

    def get_min_valid_obs(self):
        return self.__min_valid_obs


class TCA:
    def __init__(self):
        self.configuration = TCAConfig()

        self.datasets = None
        self.xn = None  # number of datasets/products
        self.xm = None  # length of each dataset/time series

    def set_datasets(self, *x):
        """
        Set input datasets for TCA.

        Parameters
        ----------
        *x : array-like
            At least three 1-D arrays with the same length.

        Notes
        -----
        Each input array should represent one product/institute/solution.
        The array length usually corresponds to time.
        """
        if len(x) < 3:
            raise ValueError("At least three datasets are required for TCA.")

        lengths = [len(xx) for xx in x]
        if len(set(lengths)) != 1:
            raise ValueError("All datasets must have the same length.")

        self.xn = len(x)
        self.xm = lengths[0]
        self.datasets = np.asarray(x, dtype=float)

        return self

    @staticmethod
    def __apply_negative_variance_policy(var_array, policy: TCANegativeVariancePolicy):
        var_array = np.asarray(var_array, dtype=float)

        # Remove tiny negative values caused by numerical round-off.
        finite_abs_max = np.nanmax(np.abs(var_array)) if np.any(np.isfinite(var_array)) else 1.0
        tol = 1e-12 * max(finite_abs_max, 1.0)

        tiny_negative = (var_array < 0) & (var_array > -tol)
        var_array[tiny_negative] = 0.0

        if policy == TCANegativeVariancePolicy.NAN:
            var_array[var_array < 0] = np.nan
        elif policy == TCANegativeVariancePolicy.CLIP:
            var_array = np.maximum(var_array, 0.0)
        elif policy == TCANegativeVariancePolicy.ABS:
            var_array = np.abs(var_array)
        else:
            raise ValueError(f"Invalid negative variance policy: {policy}")

        return var_array

    def __get_valid_covariance_matrix(self):
        """
        Construct covariance matrix after removing time samples containing NaN or Inf.
        """
        if self.datasets is None:
            raise RuntimeError("Datasets have not been set.")

        min_valid_obs = self.configuration.get_min_valid_obs()

        data = self.datasets.copy()

        valid_mask = np.all(np.isfinite(data), axis=0)
        data = data[:, valid_mask]

        if data.shape[1] < min_valid_obs:
            return None

        # De-mean each dataset.
        data = data - np.mean(data, axis=1, keepdims=True)

        # Sample covariance matrix, ddof = 1.
        cov_mat = data @ data.T / (data.shape[1] - 1)

        return cov_mat

    @staticmethod
    def __classic_tca_from_covariance(cov_mat):
        """
        Classical TCA for exactly three datasets.

        Model
        -----
        x_i = beta_i * t + epsilon_i

        Assumptions
        -----------
        Cov(t, epsilon_i) = 0
        Cov(epsilon_i, epsilon_j) = 0, i != j

        Returns
        -------
        var_error : ndarray, shape (3,)
            Error variance estimates of the three datasets.
        """
        if cov_mat.shape != (3, 3):
            raise ValueError("Classical TCA requires a 3 x 3 covariance matrix.")

        c11 = cov_mat[0, 0]
        c22 = cov_mat[1, 1]
        c33 = cov_mat[2, 2]

        c12 = cov_mat[0, 1]
        c13 = cov_mat[0, 2]
        c23 = cov_mat[1, 2]

        eps = np.finfo(float).eps

        if abs(c12) <= eps or abs(c13) <= eps or abs(c23) <= eps:
            return np.full(3, np.nan)

        signal_var_1 = c12 * c13 / c23
        signal_var_2 = c12 * c23 / c13
        signal_var_3 = c13 * c23 / c12

        var_error = np.array([
            c11 - signal_var_1,
            c22 - signal_var_2,
            c33 - signal_var_3
        ])

        return var_error

    def __run_classic(self):
        """
        Classical three-product TCA.
        """
        if self.xn != 3:
            raise ValueError("TCAMode.CLASSIC requires exactly three datasets.")

        cov_mat = self.__get_valid_covariance_matrix()
        if cov_mat is None:
            return np.full(self.xn, np.nan)

        var_array = self.__classic_tca_from_covariance(cov_mat)

        policy = self.configuration.get_negative_variance_policy()
        var_array = self.__apply_negative_variance_policy(var_array, policy)

        return var_array

    def __run_all_triplets(self):
        """
        Multiple-product TCA using all possible triplets.

        For N datasets, all C(N, 3) triplets are evaluated using classical TCA.
        For each product, all available variance estimates are aggregated by nanmedian.
        """
        cov_mat = self.__get_valid_covariance_matrix()
        if cov_mat is None:
            return np.full(self.xn, np.nan)

        var_estimates = [[] for _ in range(self.xn)]

        for comb in itertools.combinations(range(self.xn), 3):
            sub_cov = cov_mat[np.ix_(comb, comb)]
            sub_var = self.__classic_tca_from_covariance(sub_cov)

            for local_idx, global_idx in enumerate(comb):
                if np.isfinite(sub_var[local_idx]):
                    var_estimates[global_idx].append(sub_var[local_idx])

        var_array = np.full(self.xn, np.nan)

        for i in range(self.xn):
            if len(var_estimates[i]) > 0:
                var_array[i] = np.nanmedian(var_estimates[i])

        policy = self.configuration.get_negative_variance_policy()
        var_array = self.__apply_negative_variance_policy(var_array, policy)

        return var_array

    @staticmethod
    def __initial_signal_std_from_triplets(cov_mat):
        """
        Construct an initial estimate of signal standard deviation contribution
        for multiple collocation.

        The off-diagonal covariance model is:

            C_ij = q_i q_j, i != j

        where q_i = beta_i * sigma_t.
        """
        n = cov_mat.shape[0]
        signal_var_estimates = [[] for _ in range(n)]

        for comb in itertools.combinations(range(n), 3):
            sub_cov = cov_mat[np.ix_(comb, comb)]
            sub_signal_var = np.array([
                sub_cov[0, 1] * sub_cov[0, 2] / sub_cov[1, 2]
                if abs(sub_cov[1, 2]) > np.finfo(float).eps else np.nan,

                sub_cov[0, 1] * sub_cov[1, 2] / sub_cov[0, 2]
                if abs(sub_cov[0, 2]) > np.finfo(float).eps else np.nan,

                sub_cov[0, 2] * sub_cov[1, 2] / sub_cov[0, 1]
                if abs(sub_cov[0, 1]) > np.finfo(float).eps else np.nan
            ])

            for local_idx, global_idx in enumerate(comb):
                val = sub_signal_var[local_idx]
                if np.isfinite(val) and val > 0:
                    signal_var_estimates[global_idx].append(val)

        q0 = np.zeros(n)

        for i in range(n):
            if len(signal_var_estimates[i]) > 0:
                q0[i] = np.sqrt(np.nanmedian(signal_var_estimates[i]))
            else:
                q0[i] = np.sqrt(max(0.5 * cov_mat[i, i], np.finfo(float).eps))

        # Set relative signs using the first product as reference.
        # For hydrological/geodetic products measuring the same signal,
        # covariances are usually expected to be positive.
        ref = 0
        for i in range(n):
            if i == ref:
                continue

            if cov_mat[ref, i] < 0:
                q0[i] *= -1.0

        return q0

    def __run_mtc(self):
        """
        Multiple collocation by fitting the off-diagonal covariance matrix.

        Model
        -----
        C_ij = q_i q_j, i != j
        C_ii = q_i^2 + sigma_e_i^2

        where:
            q_i = beta_i * sigma_t

        Then:
            sigma_e_i^2 = C_ii - q_i^2
        """
        cov_mat = self.__get_valid_covariance_matrix()
        if cov_mat is None:
            return np.full(self.xn, np.nan)

        n = self.xn
        iu = np.triu_indices(n, k=1)

        offdiag_obs = cov_mat[iu]

        if not np.any(np.isfinite(offdiag_obs)):
            return np.full(n, np.nan)

        scale = np.nanmedian(np.abs(offdiag_obs))
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0

        q0 = self.__initial_signal_std_from_triplets(cov_mat)

        def residual(q):
            return (q[iu[0]] * q[iu[1]] - offdiag_obs) / scale

        solution = least_squares(
            residual,
            q0,
            method="trf",
            max_nfev=2000
        )

        if not solution.success:
            return np.full(n, np.nan)

        q_hat = solution.x

        signal_var = q_hat ** 2
        var_array = np.diag(cov_mat) - signal_var

        policy = self.configuration.get_negative_variance_policy()
        var_array = self.__apply_negative_variance_policy(var_array, policy)

        return var_array

    def get_variance(self):
        """
        Return TCA-based error variance estimates.
        """
        mode = self.configuration.get_mode()

        if mode == TCAMode.CLASSIC:
            var_array = self.__run_classic()
        elif mode == TCAMode.ALL_TRIPLETS:
            var_array = self.__run_all_triplets()
        elif mode == TCAMode.NLS:
            var_array = self.__run_mtc()
        else:
            raise ValueError(f"Unsupported TCA mode: {mode}")

        return var_array

    def get_std(self):
        """
        Return TCA-based error standard deviation estimates.
        """
        var_array = self.get_variance()
        return np.sqrt(var_array)


def _parse_tca_mode(mode):
    if isinstance(mode, str):
        mode = TCAMode[mode.upper()]

    if mode not in TCAMode:
        raise ValueError(f"Invalid TCA mode: {mode}")

    return mode


def _parse_negative_variance_policy(policy):
    if isinstance(policy, str):
        policy = TCANegativeVariancePolicy[policy.upper()]

    if policy not in TCANegativeVariancePolicy:
        raise ValueError(f"Invalid negative variance policy: {policy}")

    return policy


def _validate_array_inputs(*dataset):
    if len(dataset) < 3:
        raise ValueError("At least three datasets are required for TCA.")

    shapes = [np.asarray(d).shape for d in dataset]

    if len(set(shapes)) != 1:
        raise ValueError(f"All input arrays must have the same shape, got {shapes}.")

    if len(shapes[0]) < 1:
        raise ValueError("Input arrays must have at least one dimension.")

    return shapes[0]


def _tca_for_array(
        *dataset,
        mode: TCAMode,
        negative_variance_policy: TCANegativeVariancePolicy,
        min_valid_obs: int
):
    """
    TCA for numpy arrays.

    Input convention
    ----------------
    Each dataset should have shape:

        (time,)

    or

        (time, dim1, dim2, ...)

    The first axis is regarded as the sample/time dimension.
    For multi-dimensional data, TCA is performed independently at each grid cell
    or coefficient location.
    """
    mode = _parse_tca_mode(mode)
    negative_variance_policy = _parse_negative_variance_policy(negative_variance_policy)

    input_shape = _validate_array_inputs(*dataset)

    data = np.asarray(dataset, dtype=float)

    nset = data.shape[0]
    ntime = data.shape[1]

    if ntime < min_valid_obs:
        raise ValueError(
            f"Length of time/sample dimension is {ntime}, "
            f"but min_valid_obs is {min_valid_obs}."
        )

    tca_obj = TCA()
    tca_obj.configuration.set_mode(mode)
    tca_obj.configuration.set_negative_variance_policy(negative_variance_policy)
    tca_obj.configuration.set_min_valid_obs(min_valid_obs)

    # 1-D time series case: each input is shape (time,)
    if len(input_shape) == 1:
        tca_obj.set_datasets(*data)
        return tca_obj.get_std()

    # Multi-dimensional case: each input is shape (time, ...)
    spatial_shape = input_shape[1:]
    spatial_size = int(np.prod(spatial_shape))

    data_2d = data.reshape(nset, ntime, spatial_size)

    tca_results_2d = np.full((nset, spatial_size), np.nan)

    for i in range(spatial_size):
        tca_obj.set_datasets(*[data_2d[j, :, i] for j in range(nset)])
        tca_results_2d[:, i] = tca_obj.get_std()

    tca_results = tca_results_2d.reshape((nset, *spatial_shape))

    return tuple(tca_results[i] for i in range(nset))


def _tca_for_grd(
        *grid: GRD,
        mode: TCAMode,
        negative_variance_policy: TCANegativeVariancePolicy,
        min_valid_obs: int
):
    nlength = len(grid)

    dataset = [grid[i].value for i in range(nlength)]

    tca_results = _tca_for_array(
        *dataset,
        mode=mode,
        negative_variance_policy=negative_variance_policy,
        min_valid_obs=min_valid_obs
    )

    return tuple(
        GRD(tca_results[i], lat=grid[0].lat, lon=grid[0].lon)
        for i in range(nlength)
    )


def _tca_for_shc(
        *shc: SHC,
        mode: TCAMode,
        negative_variance_policy: TCANegativeVariancePolicy,
        min_valid_obs: int
):
    nlength = len(shc)

    dataset = [shc[i].value for i in range(nlength)]

    tca_results = _tca_for_array(
        *dataset,
        mode=mode,
        negative_variance_policy=negative_variance_policy,
        min_valid_obs=min_valid_obs
    )

    return tuple(
        SHC(tca_results[i])
        for i in range(nlength)
    )


def tca(
        *dataset: np.ndarray | GRD | SHC,
        mode: TCAMode | str = TCAMode.ALL_TRIPLETS,
        negative_variance_policy: TCANegativeVariancePolicy | str = TCANegativeVariancePolicy.NAN,
        min_valid_obs: int = 3
):
    """
    TCA-based error standard deviation estimation.

    Parameters
    ----------
    *dataset : np.ndarray | GRD | SHC
        At least three datasets.

        For numpy arrays, the expected shape is:

            (time,)

        or

            (time, dim1, dim2, ...)

        The first axis is interpreted as the sample/time dimension.

    mode : TCAMode | str
        TCA mode.

        TCAMode.CLASSIC:
            Classical three-product TCA. Exactly three datasets are required.

        TCAMode.ALL_TRIPLETS:
            For N >= 3 products, perform TCA for all triplets and use nanmedian.

        TCAMode.MTC:
            Multiple collocation using covariance matrix fitting.

    negative_variance_policy : TCANegativeVariancePolicy | str
        Strategy for handling negative variance estimates.

    min_valid_obs : int
        Minimum number of valid samples after removing NaN/Inf.

    Returns
    -------
    STD estimates.

    If input arrays are 1-D:
        return ndarray with shape (n_products,)

    If input arrays are multi-dimensional:
        return tuple of ndarray, one for each product.

    If input is GRD:
        return tuple of GRD.

    If input is SHC:
        return tuple of SHC.

    Notes
    -----
    The classical TCA model is:

        x_i = beta_i * t + epsilon_i

    with assumptions:

        Cov(t, epsilon_i) = 0
        Cov(epsilon_i, epsilon_j) = 0, i != j

    The returned value is the random error standard deviation estimated from TCA.
    It should not be interpreted as total uncertainty if systematic biases,
    leakage errors, filtering errors, or inter-product correlated errors exist.
    """
    if len(dataset) < 3:
        raise ValueError("At least three datasets are required for TCA.")

    mode = _parse_tca_mode(mode)
    negative_variance_policy = _parse_negative_variance_policy(negative_variance_policy)

    first_type = type(dataset[0])

    if not all(isinstance(d, first_type) for d in dataset):
        raise TypeError("All input datasets must have the same type.")

    if isinstance(dataset[0], np.ndarray):
        return _tca_for_array(
            *dataset,
            mode=mode,
            negative_variance_policy=negative_variance_policy,
            min_valid_obs=min_valid_obs
        )

    elif isinstance(dataset[0], GRD):
        return _tca_for_grd(
            *dataset,
            mode=mode,
            negative_variance_policy=negative_variance_policy,
            min_valid_obs=min_valid_obs
        )

    elif isinstance(dataset[0], SHC):
        return _tca_for_shc(
            *dataset,
            mode=mode,
            negative_variance_policy=negative_variance_policy,
            min_valid_obs=min_valid_obs
        )

    else:
        raise TypeError(f"Unsupported dataset type: {type(dataset[0])}")
