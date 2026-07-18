# src/sagea/filtering/factory.py

from __future__ import annotations

import numpy as np

from sagea.constants.constant import SHCFilterType, SHCDecorrelationType

from sagea.filtering.base import (
    infer_lmax_from_cs1d,
    cs1d_to_cs2d,
    cs2d_to_cs1d,
)

from sagea.filtering.smoothing import (
    gaussian_filter_cqlm,
    fan_filter_cqlm,
    anisotropic_gaussian_han_filter_cqlm,
)

from sagea.filtering.decorrelation import (
    pnmm_filter_cqlm,
    slide_window_filter_cqlm,
    SlideWindowMode,
)

from sagea.filtering.ddk import apply_ddk_filter_cqlm
from sagea.filtering.regularization import regularization_filter_cs
from sagea.filtering.fsc import fsc_filter_cs


def _method_name(method) -> str:
    """
    Normalize method name from enum or string.
    """
    if isinstance(method, str):
        return method.lower()

    if isinstance(method, SHCFilterType):
        return method.name.lower()

    if isinstance(method, SHCDecorrelationType):
        return method.name.lower()

    raise TypeError(f"Unsupported filtering method: {method}")


def apply_filter_to_cs(
        cs: np.ndarray,
        method,
        params: tuple | None = None,
        lmax: int | None = None,
        **kwargs,
) -> np.ndarray:
    """
    Apply filtering to 1D triangular CS coefficients.

    Parameters
    ----------
    cs : np.ndarray
        Shape:
            - (ncoef,)
            - (ntime, ncoef)
    method : str or Enum
        Supported:
            - "gaussian"
            - "fan"
            - "han"
            - "ddk"
            - "regularization"
            - "pnmm"
            - "slidewindowstable"
            - "slidewindowswenson2006"
            - "fsc"
    params : tuple, optional
        Parameters for each filter.
    lmax : int, optional
        If None, infer from cs.
    **kwargs :
        Additional keyword parameters.
    """
    cs = np.asarray(cs, dtype=float)

    if lmax is None:
        lmax = infer_lmax_from_cs1d(cs)

    name = _method_name(method)

    # ------------------------------------------------------------------
    # Regularization/FSC directly works on cs1d
    # ------------------------------------------------------------------
    if name in {"regularization", "shcfiltertype.regularization"}:
        if params is None:
            vcm_err = kwargs["vcm_err"]
            vcm_sig = kwargs["vcm_sig"]
            alpha = kwargs["alpha"]
        else:
            vcm_err, vcm_sig, alpha = params

        return regularization_filter_cs(
            cs=cs,
            vcm_err=vcm_err,
            vcm_sig=vcm_sig,
            alpha=alpha,
        )
    elif name in {"fsc", "shcfiltertype.fsc"}:
        if params is None:
            vcm_err = kwargs["vcm_err"]
            vcm_sig_list = kwargs["vcm_sig_list"]
            init_alphas = kwargs["init_alphas"]
            from_degree = kwargs["from_degree"]
            scale = kwargs["scale"]
        else:
            vcm_err, vcm_sig_list, init_alphas, from_degree, scale = params

        return fsc_filter_cs(
            cs=cs,
            vcm_err=vcm_err,
            vcm_sig_list=vcm_sig_list,
            init_alphas=init_alphas,
            from_degree=from_degree,
            scale=scale,
        )

    # ------------------------------------------------------------------
    # Other filters work on cqlm/sqlm
    # ------------------------------------------------------------------
    cqlm, sqlm, single = cs1d_to_cs2d(cs)

    # Gaussian
    if name in {"gaussian"}:
        if params is None:
            radius_km = kwargs.get("radius_km", 300.0)
        else:
            radius_km = params[0]

        cqlm_f, sqlm_f = gaussian_filter_cqlm(
            cqlm,
            sqlm,
            lmax=lmax,
            radius_km=radius_km,
        )

    # FAN
    elif name in {"fan"}:
        if params is None:
            radius1_km = kwargs.get("radius1_km", 300.0)
            radius2_km = kwargs.get("radius2_km", 300.0)
        else:
            radius1_km, radius2_km = params

        cqlm_f, sqlm_f = fan_filter_cqlm(
            cqlm,
            sqlm,
            lmax=lmax,
            radius1_km=radius1_km,
            radius2_km=radius2_km,
        )

    # HAN / anisotropic Gaussian
    elif name in {"han", "anisotropicgaussianhan"}:
        if params is None:
            radius1_km = kwargs.get("radius1_km", 300.0)
            radius2_km = kwargs.get("radius2_km", 800.0)
            m0 = kwargs.get("m0", 30)
        else:
            radius1_km, radius2_km, m0 = params

        cqlm_f, sqlm_f = anisotropic_gaussian_han_filter_cqlm(
            cqlm,
            sqlm,
            lmax=lmax,
            radius1_km=radius1_km,
            radius2_km=radius2_km,
            m0=m0,
        )

    # DDK
    elif name in {"ddk"}:
        if params is None:
            ddk_type = kwargs.get("ddk_type", 3)
        else:
            ddk_type = params[0]

        cqlm_f, sqlm_f = apply_ddk_filter_cqlm(
            cqlm,
            sqlm,
            ddk_type=ddk_type,
        )

    # PnMm
    elif name in {"pnmm"}:
        if params is None:
            poly_n = kwargs.get("poly_n", 3)
            start_m = kwargs.get("start_m", 10)
        else:
            poly_n, start_m = params

        cqlm_f, sqlm_f = pnmm_filter_cqlm(
            cqlm,
            sqlm,
            poly_n=poly_n,
            start_m=start_m,
        )

    # SlideWindow Stable
    elif name in {"slidewindowstable", "slide_window_stable"}:
        if params is None:
            poly_n = kwargs.get("poly_n", 3)
            start_m = kwargs.get("start_m", 10)
            window_length = kwargs.get("window_length", 10)
        else:
            poly_n, start_m, window_length = params

        cqlm_f, sqlm_f = slide_window_filter_cqlm(
            cqlm,
            sqlm,
            poly_n=poly_n,
            start_m=start_m,
            window_length=window_length,
            mode=SlideWindowMode.Stable,
        )

    # SlideWindow Swenson/Wahr 2006
    elif name in {
        "slidewindowswenson2006",
        "slidewindowwahr2006",
        "slide_window_wahr2006",
        "slide_window_swenson2006",
    }:
        if params is None:
            poly_n = kwargs.get("poly_n", 3)
            start_m = kwargs.get("start_m", 10)
            window_length = kwargs.get("window_length", 10)
            a = kwargs.get("a", 30.0)
            k = kwargs.get("k", 5.0)
        else:
            poly_n, start_m, window_length, a, k = params

        cqlm_f, sqlm_f = slide_window_filter_cqlm(
            cqlm,
            sqlm,
            poly_n=poly_n,
            start_m=start_m,
            window_length=window_length,
            mode=SlideWindowMode.Wahr2006,
            a=a,
            k=k,
        )

    else:
        raise ValueError(f"Unsupported filtering method: {method}")

    return cs2d_to_cs1d(
        cqlm_f,
        sqlm_f,
        single=single,
    )
