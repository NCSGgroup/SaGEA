# src/sagea/filtering/decorrelation.py

from __future__ import annotations

from enum import Enum

import numpy as np

from sagea.filtering.base import (
    ensure_cs_3d,
    restore_cs_dimension,
    polynomial_design_matrix,
)


class SlideWindowMode(Enum):
    Stable = "stable"
    Wahr2006 = "wahr2006"


def _pnmm_apply_to_cqlm(
    cqlm: np.ndarray,
    poly_n: int = 3,
    start_m: int = 10,
) -> np.ndarray:
    """
    PnMm decorrelation for one C/S-like array.

    Parameters
    ----------
    cqlm : np.ndarray
        Shape: (ntime, lmax + 1, lmax + 1)
    """
    cqlm = np.asarray(cqlm, dtype=float).copy()

    lmax = cqlm.shape[1] - 1

    def get_fit_array(array: np.ndarray) -> np.ndarray:
        t = np.arange(array.shape[0]) + 1
        A = polynomial_design_matrix(t, poly_n)
        fit_params = np.linalg.pinv(A) @ array
        fit_array = A @ fit_params
        return fit_array

    for m in range(start_m, lmax + 1):
        l_even_like = np.arange(m, lmax + 1, 2)
        l_odd_like = np.arange(m + 1, lmax + 1, 2)

        if len(l_even_like) <= poly_n or len(l_odd_like) <= poly_n:
            continue

        array1 = cqlm[:, l_even_like, m].T
        array2 = cqlm[:, l_odd_like, m].T

        array1 = array1 - get_fit_array(array1)
        array2 = array2 - get_fit_array(array2)

        cqlm[:, l_even_like, m] = array1.T
        cqlm[:, l_odd_like, m] = array2.T

    return cqlm


def pnmm_filter_cqlm(
    cqlm: np.ndarray,
    sqlm: np.ndarray,
    poly_n: int = 3,
    start_m: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    PnMm decorrelation filtering.

    This function processes arrays only and does not depend on SHC.
    """
    cqlm3, sqlm3, single = ensure_cs_3d(cqlm, sqlm)

    ntime = cqlm3.shape[0]

    csqlm = np.concatenate([cqlm3, sqlm3], axis=0)
    csqlm_f = _pnmm_apply_to_cqlm(
        csqlm,
        poly_n=poly_n,
        start_m=start_m,
    )

    cqlm_f = csqlm_f[:ntime]
    sqlm_f = csqlm_f[ntime:]

    return restore_cs_dimension(cqlm_f, sqlm_f, single)


def _slide_window_decorrelation_for_array(
    array: np.ndarray,
    window_length: int,
    poly_n: int,
    q: int,
) -> np.ndarray:
    """
    Parameters
    ----------
    array : np.ndarray
        Shape: (ntime, n_degree)
    """
    array = array.T.copy()

    t = np.arange(window_length)
    design_mat = polynomial_design_matrix(t, poly_n)

    array_expand = None

    for m_pie in range(0, array.shape[0] - window_length + 1):
        part = array[m_pie: window_length + m_pie, :]

        if array_expand is None:
            array_expand = part
        else:
            array_expand = np.concatenate([array_expand, part], axis=1)

    if array_expand is None:
        return array.T

    fit_params = np.linalg.pinv(design_mat) @ array_expand
    fit_array = design_mat @ fit_params

    array_expand = array_expand - fit_array

    if array_expand.shape[1] == q:
        array_new = array_expand
    else:
        half = int(window_length / 2)

        array_new_first = array_expand[: half + 1, :q]
        array_new_last = array_expand[half:, -q:]
        array_new_middle = array_expand[half, q:-q].reshape((-1, q))

        array_new = np.concatenate(
            [array_new_first, array_new_middle, array_new_last],
            axis=0,
        )

    return array_new.T


def _slide_window_apply_to_cqlm(
    cqlm: np.ndarray,
    poly_n: int = 3,
    start_m: int = 10,
    window_length: int = 5,
    mode: SlideWindowMode = SlideWindowMode.Stable,
    a: float = 10.0,
    k: float = 30.0,
) -> np.ndarray:
    """
    Sliding-window decorrelation for one C/S-like array.

    Parameters
    ----------
    cqlm : np.ndarray
        Shape: (ntime, lmax + 1, lmax + 1)
    """
    cqlm = np.asarray(cqlm, dtype=float).copy()

    lmax = cqlm.shape[1] - 1
    q = cqlm.shape[0]

    if isinstance(mode, str):
        mode = SlideWindowMode(mode)

    if mode is SlideWindowMode.Wahr2006:
        wl = np.trunc(a * np.exp(-np.arange(lmax + 1) / k) + 1)
        wl[wl < window_length] = window_length
        wl += wl % 2 - 1
    elif mode is SlideWindowMode.Stable:
        wl = np.ones(lmax + 1) * window_length
    else:
        raise ValueError(f"Unsupported slide window mode: {mode}")

    for m in range(start_m, lmax + 1):
        this_window_length = int(wl[m])

        if this_window_length < poly_n + 1:
            raise ValueError(
                "window_length should be larger than polynomial degree."
            )

        array_even = cqlm[:, m + 1::2, m]
        array_odd = cqlm[:, m::2, m]

        if array_even.shape[1] >= this_window_length:
            cqlm[:, m + 1::2, m] = _slide_window_decorrelation_for_array(
                array_even,
                window_length=this_window_length,
                poly_n=poly_n,
                q=q,
            )

        if array_odd.shape[1] >= this_window_length:
            cqlm[:, m::2, m] = _slide_window_decorrelation_for_array(
                array_odd,
                window_length=this_window_length,
                poly_n=poly_n,
                q=q,
            )

    return cqlm


def slide_window_filter_cqlm(
    cqlm: np.ndarray,
    sqlm: np.ndarray,
    poly_n: int = 3,
    start_m: int = 10,
    window_length: int = 5,
    mode: SlideWindowMode | str = SlideWindowMode.Stable,
    a: float = 10.0,
    k: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sliding-window decorrelation filtering.
    """
    cqlm3, sqlm3, single = ensure_cs_3d(cqlm, sqlm)

    ntime = cqlm3.shape[0]

    csqlm = np.concatenate([cqlm3, sqlm3], axis=0)

    csqlm_f = _slide_window_apply_to_cqlm(
        csqlm,
        poly_n=poly_n,
        start_m=start_m,
        window_length=window_length,
        mode=mode,
        a=a,
        k=k,
    )

    cqlm_f = csqlm_f[:ntime]
    sqlm_f = csqlm_f[ntime:]

    return restore_cs_dimension(cqlm_f, sqlm_f, single)