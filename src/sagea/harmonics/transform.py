from __future__ import annotations

import numpy as np

from sagea.harmonics.harmonic import Harmonic, GRDType
from sagea.utils.MathTool import MathTool


def _as_3d_grid(grid: np.ndarray) -> tuple[np.ndarray, bool]:
    grid = np.asarray(grid, dtype=float)

    if grid.ndim == 2:
        return grid[None, :, :], True

    if grid.ndim == 3:
        return grid, False

    raise ValueError(f"grid should be 2D or 3D, got shape {grid.shape}.")


def _as_3d_cs(
        cqlm: np.ndarray,
        sqlm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool]:

    cqlm = np.asarray(cqlm, dtype=float)
    sqlm = np.asarray(sqlm, dtype=float)

    if cqlm.shape != sqlm.shape:
        raise ValueError(
            f"cqlm and sqlm shape mismatch: {cqlm.shape} vs {sqlm.shape}."
        )

    if cqlm.ndim == 2:
        return cqlm[None, :, :], sqlm[None, :, :], True

    if cqlm.ndim == 3:
        return cqlm, sqlm, False

    raise ValueError(
        f"cqlm/sqlm should be 2D or 3D, got shape {cqlm.shape}."
    )


def _make_harmonic(
        lmax: int,
        grid_type: GRDType | None = GRDType.GLQ,
        grid_space: float | None = None,
) -> tuple[Harmonic, np.ndarray, np.ndarray]:
    """
    Create Harmonic instance and return geographic lat/lon in degrees.

    Rules
    -----
    1. If grid_space is given, use custom regular lat/lon grid.
       Then grid_type must be None internally.
    2. If grid_space is None, use fixed quadrature grid_type.
    """


    if grid_space is not None:
        lat, lon = MathTool.get_global_lat_lon_range(grid_space)

        har = Harmonic(
            lmax=lmax,
            grid_type=None,
            lat=lat,
            lon=lon,
        )

        return har, np.asarray(lat), np.asarray(lon)

    if grid_type is None:
        raise ValueError("Either grid_space or grid_type should be provided.")

    har = Harmonic(
        lmax=lmax,
        grid_type=grid_type,
        lat=None,
        lon=None,
    )

    lat = 90.0 - np.rad2deg(har.colat)
    lon = np.rad2deg(har.lon)

    return har, lat, lon


def shc_to_grid(
        cqlm: np.ndarray,
        sqlm: np.ndarray,
        lmax: int | None = None,
        grid_type: GRDType | None = GRDType.GLQ,
        grid_space: float | None = None,
        factor: np.ndarray | None = None,
):
    """
    SH coefficients -> grid.
    """

    cqlm, sqlm, single = _as_3d_cs(cqlm, sqlm)

    if lmax is None:
        lmax = cqlm.shape[1] - 1

    cqlm = cqlm[:, :lmax + 1, :lmax + 1].copy()
    sqlm = sqlm[:, :lmax + 1, :lmax + 1].copy()

    if factor is not None:
        factor = np.asarray(factor, dtype=float)

        if factor.shape != (lmax + 1,):
            raise ValueError(
                f"factor should have shape {(lmax + 1,)}, got {factor.shape}."
            )

        cqlm *= factor[None, :, None]
        sqlm *= factor[None, :, None]

    har, lat, lon = _make_harmonic(
        lmax=lmax,
        grid_type=grid_type,
        grid_space=grid_space,
    )

    grid = har.synthesis(cqlm, sqlm)

    if single:
        grid = grid[0]

    return grid, lat, lon


def grid_to_shc(
        grid: np.ndarray,
        lmax: int,
        grid_type: GRDType | None = GRDType.GLQ,
        grid_space: float | None = None,
        factor: np.ndarray | None = None,
):
    """
    Grid -> SH coefficients.
    """

    grid, single = _as_3d_grid(grid)

    har, lat, lon = _make_harmonic(
        lmax=lmax,
        grid_type=grid_type,
        grid_space=grid_space,
    )

    cqlm, sqlm = har.analysis(grid)

    if single:
        cqlm = cqlm[None, :, :]
        sqlm = sqlm[None, :, :]

    if factor is not None:
        factor = np.asarray(factor, dtype=float)

        if factor.shape != (lmax + 1,):
            raise ValueError(
                f"factor should have shape {(lmax + 1,)}, got {factor.shape}."
            )

        cqlm *= factor[None, :, None]
        sqlm *= factor[None, :, None]

    if single:
        return cqlm[0], sqlm[0]

    return cqlm, sqlm