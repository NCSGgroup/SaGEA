# src/sagea/filtering/regularization.py

from __future__ import annotations

import numpy as np

from sagea.filtering.base import cs2d_to_cs1d, cs1d_to_cs2d


def regularization_filter_cs(
        cs: np.ndarray,
        vcm_err: np.ndarray,
        vcm_sig: np.ndarray,
        alpha: float,
        from_degree: int = 0,
) -> np.ndarray:
    """
    Regularization filter for 1D triangular CS coefficients.

    Parameters
    ----------
    cs : np.ndarray
        Shape:
            - (ncoef,)
            - (ntime, ncoef)
    vcm_err : np.ndarray
        Error variance-covariance matrix.
    vcm_sig : np.ndarray
        Signal variance-covariance matrix.
    alpha : float
        Regularization parameter.
    from_degree : int
    """
    cs = np.asarray(cs, dtype=float)

    single = cs.ndim == 1

    if single:
        cs2 = cs[None, :]
    elif cs.ndim == 2:
        cs2 = cs
    else:
        raise ValueError(f"cs should be 1D or 2D, got shape {cs.shape}.")

    c_mat = np.asarray(vcm_err, dtype=float)[from_degree ** 2:, from_degree ** 2:]
    d_mat = np.asarray(vcm_sig, dtype=float)[from_degree ** 2:, from_degree ** 2:]

    a_c_dinv = alpha * np.linalg.solve(d_mat.T, c_mat.T).T
    eye = np.eye(c_mat.shape[0])

    cs_filtered = np.zeros_like(cs2)
    cs_filtered[:, :from_degree ** 2] = cs2[:, :from_degree ** 2]
    cs_filtered[:, from_degree ** 2:] = np.linalg.solve(eye + a_c_dinv, cs2.T).T

    return cs_filtered[0] if single else cs_filtered


def regularization_filter_cqlm(
        cqlm: np.ndarray,
        sqlm: np.ndarray,
        vcm_err: np.ndarray,
        vcm_sig: np.ndarray,
        alpha: float,
):
    """
    Regularization filter for C/S matrices.
    """
    single = cqlm.ndim == 2

    cs = cs2d_to_cs1d(cqlm, sqlm, single=single)

    cs_filtered = regularization_filter_cs(
        cs=cs,
        vcm_err=vcm_err,
        vcm_sig=vcm_sig,
        alpha=alpha,
    )

    cqlm_f, sqlm_f, _ = cs1d_to_cs2d(cs_filtered)

    return cqlm_f, sqlm_f
