# src/sagea/filtering/base.py

from __future__ import annotations

import numpy as np

from sagea.utils import MathTool


def ensure_cs_3d(
    cqlm: np.ndarray,
    sqlm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Ensure C/S coefficient arrays are 3D.

    Parameters
    ----------
    cqlm, sqlm : np.ndarray
        Shape:
            - (lmax + 1, lmax + 1)
            - (ntime, lmax + 1, lmax + 1)

    Returns
    -------
    cqlm3, sqlm3, single
    """
    cqlm = np.asarray(cqlm, dtype=float)
    sqlm = np.asarray(sqlm, dtype=float)

    if cqlm.shape != sqlm.shape:
        raise ValueError(
            f"cqlm and sqlm shape mismatch: {cqlm.shape} vs {sqlm.shape}"
        )

    if cqlm.ndim == 2:
        return cqlm[None, :, :], sqlm[None, :, :], True

    if cqlm.ndim == 3:
        return cqlm, sqlm, False

    raise ValueError(
        f"cqlm/sqlm should be 2D or 3D, got ndim={cqlm.ndim}."
    )


def restore_cs_dimension(
    cqlm: np.ndarray,
    sqlm: np.ndarray,
    single: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if single:
        if cqlm.shape[0] != 1 or sqlm.shape[0] != 1:
            raise ValueError("Invalid single-epoch C/S arrays.")
        return cqlm[0], sqlm[0]

    return cqlm, sqlm


def infer_lmax_from_cs1d(cs: np.ndarray) -> int:
    cs = np.asarray(cs)

    if cs.ndim == 1:
        ncoef = cs.shape[0]
    elif cs.ndim == 2:
        ncoef = cs.shape[1]
    else:
        raise ValueError(f"cs should be 1D or 2D, got shape {cs.shape}.")

    lmax_float = np.sqrt(ncoef) - 1
    lmax = int(round(lmax_float))

    if not np.isclose(lmax_float, lmax, atol=1e-8):
        raise ValueError(
            f"Invalid coefficient length {ncoef}. Expected (lmax + 1)^2."
        )

    return lmax


def cs1d_to_cs2d(cs: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Convert 1D triangular CS array to C/S 2D matrix.

    Parameters
    ----------
    cs : np.ndarray
        Shape:
            - (ncoef,)
            - (ntime, ncoef)

    Returns
    -------
    cqlm, sqlm, single
        If input is 1D, cqlm/sqlm are 2D and single=True.
        If input is 2D, cqlm/sqlm are 3D and single=False.
    """
    cs = np.asarray(cs, dtype=float)

    if cs.ndim == 1:
        cqlm, sqlm = MathTool.cs_decompose_triangle1d_to_cs2d(cs)
        return cqlm, sqlm, True

    if cs.ndim == 2:
        c_list = []
        s_list = []

        for item in cs:
            c, s = MathTool.cs_decompose_triangle1d_to_cs2d(item)
            c_list.append(c)
            s_list.append(s)

        return np.asarray(c_list), np.asarray(s_list), False

    raise ValueError(f"cs should be 1D or 2D, got shape {cs.shape}.")


def cs2d_to_cs1d(
    cqlm: np.ndarray,
    sqlm: np.ndarray,
    single: bool = False,
) -> np.ndarray:
    """
    Convert C/S matrices to 1D triangular CS array.
    """
    cqlm = np.asarray(cqlm, dtype=float)
    sqlm = np.asarray(sqlm, dtype=float)

    if cqlm.shape != sqlm.shape:
        raise ValueError(
            f"cqlm and sqlm shape mismatch: {cqlm.shape} vs {sqlm.shape}"
        )

    if cqlm.ndim == 2:
        return MathTool.cs_combine_to_triangle_1d(cqlm, sqlm)

    if cqlm.ndim == 3:
        out = [
            MathTool.cs_combine_to_triangle_1d(cqlm[i], sqlm[i])
            for i in range(cqlm.shape[0])
        ]
        out = np.asarray(out)
        return out[0] if single else out

    raise ValueError(f"cqlm/sqlm should be 2D or 3D, got ndim={cqlm.ndim}.")


def gaussian_weight_1d(
    lmax: int,
    radius_smooth: float,
    radius_earth: float,
) -> np.ndarray:
    """
    Gaussian smoothing weight.

    Parameters
    ----------
    lmax : int
    radius_smooth : float
        Unit: meter.
    radius_earth : float
        Unit: meter.
    """
    w = np.zeros(lmax + 1)

    if radius_smooth == 0:
        return np.ones(lmax + 1)

    b = np.log(2.0) / (1.0 - np.cos(radius_smooth / radius_earth))

    w[0] = 1.0

    if lmax == 0:
        return w

    w[1] = (1.0 + np.exp(-2.0 * b)) / (1.0 - np.exp(-2.0 * b)) - 1.0 / b

    for i in range(1, lmax):
        w[i + 1] = -(w[i] * (2 * i + 1)) / b + w[i - 1]

    return w


def polynomial_design_matrix(t: np.ndarray, degree: int) -> np.ndarray:
    """
    Polynomial design matrix:

        A[:, p] = t ** p
    """
    t = np.asarray(t, dtype=float)
    return np.vstack([t ** p for p in range(degree + 1)]).T