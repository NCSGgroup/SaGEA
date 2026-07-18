# src/sagea/filtering/smoothing.py

from __future__ import annotations

import numpy as np

from sagea.constants.constant import GeoConstant
from sagea.utils import MathTool
from sagea.filtering.base import (
    ensure_cs_3d,
    restore_cs_dimension,
    gaussian_weight_1d,
)


def gaussian_weight_matrix(
    lmax: int,
    radius_km: float = 300.0,
    earth_radius: float = GeoConstant.radius_earth,
) -> np.ndarray:
    """
    Isotropic Gaussian weight matrix W_lm.

    Weight depends only on degree l.
    """
    radius_m = radius_km * 1000.0

    w = gaussian_weight_1d(
        lmax=lmax,
        radius_smooth=radius_m,
        radius_earth=earth_radius,
    )

    wlm = np.tile(w, (lmax + 1, 1)).T

    return np.tril(wlm)


def fan_weight_matrix(
    lmax: int,
    radius1_km: float = 300.0,
    radius2_km: float = 300.0,
    earth_radius: float = GeoConstant.radius_earth,
) -> np.ndarray:
    """
    FAN filter weight matrix.
    """
    r1 = radius1_km * 1000.0
    r2 = radius2_km * 1000.0

    w1 = gaussian_weight_1d(lmax, r1, earth_radius)
    w2 = gaussian_weight_1d(lmax, r2, earth_radius)

    matrix = w1[:, None] @ w2[None, :]

    return np.tril(matrix)


def anisotropic_gaussian_han_weight_matrix(
    lmax: int,
    radius1_km: float = 300.0,
    radius2_km: float = 800.0,
    m0: int = 30,
    earth_radius: float = GeoConstant.radius_earth,
) -> np.ndarray:
    """
    Anisotropic Gaussian filter proposed by Han-type formulation.

    The smoothing radius changes with order m.
    """
    r1 = radius1_km * 1000.0
    r2 = radius2_km * 1000.0

    matrix = np.zeros((lmax + 1, lmax + 1))

    for m in range(lmax + 1):
        radius_m = (r2 - r1) / m0 * m + r1
        w = gaussian_weight_1d(lmax, radius_m, earth_radius)
        matrix[:, m] = w

    return np.tril(matrix)


def apply_weight_matrix_to_cqlm(
    cqlm: np.ndarray,
    sqlm: np.ndarray,
    weight_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply degree/order weight matrix to C/S coefficients.
    """
    cqlm3, sqlm3, single = ensure_cs_3d(cqlm, sqlm)

    cqlm_f = cqlm3 * weight_matrix
    sqlm_f = sqlm3 * weight_matrix

    return restore_cs_dimension(cqlm_f, sqlm_f, single)


def gaussian_filter_cqlm(
    cqlm: np.ndarray,
    sqlm: np.ndarray,
    lmax: int,
    radius_km: float = 300.0,
    earth_radius: float = GeoConstant.radius_earth,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply isotropic Gaussian filtering to C/S matrices.
    """
    if radius_km == 0:
        return cqlm, sqlm

    weight = gaussian_weight_matrix(
        lmax=lmax,
        radius_km=radius_km,
        earth_radius=earth_radius,
    )

    return apply_weight_matrix_to_cqlm(cqlm, sqlm, weight)


def fan_filter_cqlm(
    cqlm: np.ndarray,
    sqlm: np.ndarray,
    lmax: int,
    radius1_km: float = 300.0,
    radius2_km: float = 300.0,
    earth_radius: float = GeoConstant.radius_earth,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply FAN filtering to C/S matrices.
    """
    weight = fan_weight_matrix(
        lmax=lmax,
        radius1_km=radius1_km,
        radius2_km=radius2_km,
        earth_radius=earth_radius,
    )

    return apply_weight_matrix_to_cqlm(cqlm, sqlm, weight)


def anisotropic_gaussian_han_filter_cqlm(
    cqlm: np.ndarray,
    sqlm: np.ndarray,
    lmax: int,
    radius1_km: float = 300.0,
    radius2_km: float = 800.0,
    m0: int = 30,
    earth_radius: float = GeoConstant.radius_earth,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply anisotropic Gaussian Han filtering to C/S matrices.
    """
    weight = anisotropic_gaussian_han_weight_matrix(
        lmax=lmax,
        radius1_km=radius1_km,
        radius2_km=radius2_km,
        m0=m0,
        earth_radius=earth_radius,
    )

    return apply_weight_matrix_to_cqlm(cqlm, sqlm, weight)


def gaussian_weight_cs1d(
    lmax: int,
    radius_km: float = 300.0,
    earth_radius: float = GeoConstant.radius_earth,
) -> np.ndarray:
    weight = gaussian_weight_matrix(lmax, radius_km, earth_radius)
    return MathTool.cs_combine_to_triangle_1d(weight, weight)


def fan_weight_cs1d(
    lmax: int,
    radius1_km: float = 300.0,
    radius2_km: float = 300.0,
    earth_radius: float = GeoConstant.radius_earth,
) -> np.ndarray:
    weight = fan_weight_matrix(lmax, radius1_km, radius2_km, earth_radius)
    return MathTool.cs_combine_to_triangle_1d(weight, weight)


def anisotropic_gaussian_han_weight_cs1d(
    lmax: int,
    radius1_km: float = 300.0,
    radius2_km: float = 800.0,
    m0: int = 30,
    earth_radius: float = GeoConstant.radius_earth,
) -> np.ndarray:
    weight = anisotropic_gaussian_han_weight_matrix(
        lmax,
        radius1_km,
        radius2_km,
        m0,
        earth_radius,
    )
    return MathTool.cs_combine_to_triangle_1d(weight, weight)