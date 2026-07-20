# src/sagea/filtering/fsc.py

from __future__ import annotations

import sys

import numpy as np

from sagea.filtering.base import cs2d_to_cs1d, cs1d_to_cs2d


def bayesian_filter_vce(
        x, C, *D_matrices,
        init_alphas=None,
        tol=1e-4,
        max_iter=100,
        verbose=True,
        scale=1.
):
    """
    S = a1*D1 + a2*D2 + ..., x_hat = (C^-1 + S^-1)^-1 C^-1 x

    tol: Relative convergence tolerance. The iteration stops when all alpha rate changes are less than tol.
    max_iter: maximum number of iterations。
    """
    N = len(x)
    num_priors = len(D_matrices)

    if init_alphas is None:
        alphas = np.ones(num_priors)
    else:
        alphas = np.asarray(init_alphas, dtype=float)

    if verbose:
        print("Starting VCE Iterations...")

    alpha_list = []

    iteration = 0
    for iteration in range(1, max_iter + 1):
        # 1. W = S + C
        S = np.zeros((N, N))
        for a, D_mat in zip(alphas, D_matrices):
            S += a * D_mat
        W = S + C

        # inverse of W
        try:
            inv_W = np.linalg.inv(W)
        except np.linalg.LinAlgError:
            raise ValueError("Matrix W became singular during VCE iteration.")

        # u = W^{-1} x
        u = inv_W @ x

        new_alphas = np.zeros(num_priors)

        # 2. VCE to determine alpha
        for i, D_mat in enumerate(D_matrices):
            numerator = u.T @ D_mat @ u
            denominator = np.trace(inv_W @ D_mat)
            if denominator > 0:
                new_alphas[i] = alphas[i] * (numerator / denominator)
            else:
                new_alphas[i] = alphas[i]

        # 3. check convergence
        # diff = np.abs(new_alphas - alphas) / (alphas + 1e-12)
        diff = np.abs(new_alphas - alphas) / alphas
        max_diff = np.max(diff)

        if verbose:
            sys.stdout.write(f"\r[VCE] Iter: {iteration:03d} | "
                             f"Max Diff: {max_diff:.4e} | Alphas: {new_alphas}")
            sys.stdout.flush()

        alpha_list.append(alphas)
        alphas = new_alphas

        if max_diff < tol:
            if verbose:
                print(f"\n-> VCE Converged successfully in {iteration} iterations.")
            break

    if iteration == max_iter and verbose:
        print(f"\n-> Warning: VCE reached max_iter ({max_iter}) without strict convergence.")

    # x_hat = S(S+C)^-1 x = x - C(S+C)^-1 x
    S_final = np.zeros((N, N))
    for a, D_mat in zip(alphas, D_matrices):
        S_final += a * D_mat

    inv_W_final = np.linalg.inv((1 / scale) * S_final + C)

    x_hat = x - C @ (inv_W_final @ x)

    return x_hat


def fsc_filter_cs(
        cs: np.ndarray,
        vcm_err: np.ndarray,
        vcm_sig_list: list[np.ndarray],
        init_alphas=None,
        from_degree=2,
        scale: float = 1.0,
        tol=1e-4,
        max_iter=100,
        verbose=True,

) -> np.ndarray:
    """
    Regularization filter for 1D triangular CS coefficients.

    Parameters
    ----------
    cs : np.ndarray
        Shape:
            - (ncoef,)
            - (ntime, ncoef) # not supported for multi sets yet
    vcm_err : np.ndarray
        Error variance-covariance matrix.
    vcm_sig_list : list of np.ndarray
        Signal variance-covariance matrix.
    init_alphas : list of float
        initial parameter of FSC filter for each vcm_sig.
    from_degree : filtering from degree, default 2
    scale : float, default 1.0,
    tol : float, default 1e-4
    max_iter : int, default 100
    verbose : bool, default True
    """
    cs = np.asarray(cs, dtype=float)

    single = cs.ndim == 1

    if single:
        cs2 = cs[None, :]
    elif cs.ndim == 2:
        cs2 = cs
    else:
        raise ValueError(f"cs should be 1D or 2D, got shape {cs.shape}.")

    if cs2.shape[0] > 1:
        raise ValueError(f"cs should be in length 1, got length {cs2.shape[0]}.")

    c_mat = np.asarray(vcm_err, dtype=float)[from_degree ** 2:, from_degree ** 2:]
    d_mat = [
        np.asarray(vcm_sig_list[i], dtype=float)[from_degree ** 2:, from_degree ** 2:]
        for i in range(len(vcm_sig_list))
    ]
    cs_filtered = np.zeros_like(cs2)
    cs_filtered[:, :from_degree ** 2] = cs2[:, :from_degree ** 2]

    cs_filtered[0, from_degree ** 2:] = bayesian_filter_vce(
        cs[0, from_degree ** 2:],
        c_mat,
        *d_mat,
        init_alphas=init_alphas,
        scale=scale,
        tol=tol,
        max_iter=max_iter,
        verbose=verbose,
    )

    return cs_filtered


def fsc_filter_cqlm(
        cqlm: np.ndarray,
        sqlm: np.ndarray,
        vcm_err: np.ndarray,
        vcm_sig_list: list[np.ndarray],
        init_alphas=None,
        from_degree=2,
        scale: float = 1.0,
):
    """
    Regularization filter for C/S matrices.
    """
    single = cqlm.ndim == 2

    cs = cs2d_to_cs1d(cqlm, sqlm, single=single)

    cs_filtered = fsc_filter_cs(
        cs=cs,
        vcm_err=vcm_err,
        vcm_sig_list=vcm_sig_list,
        init_alphas=init_alphas,
        from_degree=from_degree,
        scale=scale,
    )

    cqlm_f, sqlm_f, _ = cs1d_to_cs2d(cs_filtered)

    return cqlm_f, sqlm_f
