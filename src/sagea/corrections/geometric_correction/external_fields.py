#!/use/bin/env python
# coding=utf-8
# @Author  : Shuhao Liu
# @Time    : 2026/6/18 18:05 
# @File    : external_fields.py
from __future__ import annotations

import numpy as np
from netCDF4 import Dataset
from scipy.interpolate import RegularGridInterpolator

from sagea.harmonics.harmonic import Harmonic
from sagea.corrections.geometric_correction.LoadSH import Gif48
from sagea.corrections.geometric_correction.RefEllipsoid import (
    RefEllipsoid,
    EllipsoidType,
)


def _triangular_1d_to_lm(
        cs: np.ndarray,
        lmax: int,
) -> np.ndarray:
    """
    Convert old 1D triangular CS array to 2D lm matrix.

    Old ordering:
        index = l * (l + 1) / 2 + m

    Returns
    -------
    arr : ndarray
        Shape (lmax + 1, lmax + 1)
    """

    cs = np.asarray(cs, dtype=float)

    expected = (lmax + 1) * (lmax + 2) // 2

    if len(cs) < expected:
        raise ValueError(
            f"CS length should be at least {expected}, got {len(cs)}."
        )

    arr = np.zeros((lmax + 1, lmax + 1), dtype=float)

    k = 0
    for l in range(lmax + 1):
        arr[l, :l + 1] = cs[k:k + l + 1]
        k += l + 1

    return arr


def read_era5_surface_geopotential(
        nc_file: str,
        target_lat: np.ndarray,
        target_lon: np.ndarray,
        variable: str = "z",
        time_index: int = 0,
) -> np.ndarray:
    """
    Read ERA5 surface geopotential PHISFC and interpolate to target grid.

    Parameters
    ----------
    nc_file:
        Path to PHISFC_ERA5_invariant.nc.

    target_lat, target_lon:
        Target latitude/longitude in degrees.

    variable:
        ERA5 surface geopotential variable name.
        Usually "z".

    time_index:
        Usually 0 for invariant field.

    Returns
    -------
    orography:
        Surface geopotential on target grid.
        Unit is usually m^2 / s^2.

        Shape:
            (nlat, nlon)
    """

    target_lat = np.asarray(target_lat, dtype=float)
    target_lon = np.asarray(target_lon, dtype=float)

    with Dataset(nc_file) as nc:
        lat_src = np.asarray(nc.variables["latitude"][:], dtype=float)
        lon_src = np.asarray(nc.variables["longitude"][:], dtype=float)

        z = np.asarray(nc.variables[variable][time_index, :, :], dtype=float)

    # ------------------------------------------------------------
    # Ensure latitude is ascending for interpolation.
    # ERA5 latitude is often descending: 90 -> -90.
    # ------------------------------------------------------------
    lat_order = np.argsort(lat_src)
    lat_src = lat_src[lat_order]
    z = z[lat_order, :]

    # ------------------------------------------------------------
    # Normalize source longitude to [0, 360).
    # ------------------------------------------------------------
    lon_src = np.mod(lon_src, 360.0)
    lon_order = np.argsort(lon_src)
    lon_src = lon_src[lon_order]
    z = z[:, lon_order]

    # ------------------------------------------------------------
    # Periodic extension in longitude.
    # ------------------------------------------------------------
    lon_ext = np.concatenate([lon_src, [lon_src[0] + 360.0]])
    z_ext = np.concatenate([z, z[:, :1]], axis=1)

    interpolator = RegularGridInterpolator(
        points=(lat_src, lon_ext),
        values=z_ext,
        bounds_error=False,
        fill_value=None,
    )

    lon_target = np.mod(target_lon, 360.0)

    lat2d, lon2d = np.meshgrid(
        target_lat,
        lon_target,
        indexing="ij",
    )

    points = np.column_stack([
        lat2d.ravel(),
        lon2d.ravel(),
    ])

    out = interpolator(points).reshape(
        len(target_lat),
        len(target_lon),
    )

    return out


def compute_geoid_undulation_from_gif48(
        gif48_file: str,
        target_lat: np.ndarray,
        target_lon: np.ndarray,
        geoid_lmax: int = 160,
        ellipsoid_type: EllipsoidType = EllipsoidType.GRS80_IERS2010,
) -> np.ndarray:
    """
    Compute geoid undulation on target grid using GIF48 gravity model.

    This is the vectorized replacement of old:

        GeoidUndulation(elltype).getGeoid(lat, lon)

    Old logic:
        C, S = GIF48.getCS(Nmax)
        C[0:3] = 0
        S[0:3] = 0
        C -= normal gravity coefficients
        geoid = Harmonic.synthesis(..., SynthesisType.Geoidheight)

    In old Harmonic:
        Geoidheight synthesis factor =
            GM / a / je

    Parameters
    ----------
    gif48_file:
        Path to GIF48.gfc.

    target_lat, target_lon:
        Target grid in degrees.

    geoid_lmax:
        Degree used for geoid undulation.

    ellipsoid_type:
        Reference ellipsoid.

    Returns
    -------
    undulation:
        Geoid undulation in meters.
        Shape (nlat, nlon)
    """

    target_lat = np.asarray(target_lat, dtype=float)
    target_lon = np.asarray(target_lon, dtype=float)

    ellipsoid = RefEllipsoid(ellipsoid_type)

    sh = Gif48().load(gif48_file)

    c1d, s1d = sh.getCS(geoid_lmax)

    c1d = np.asarray(c1d, dtype=float)
    s1d = np.asarray(s1d, dtype=float)

    # Same as old GeoidUndulation.getGeoid()
    c1d[0:3] = 0.0
    s1d[0:3] = 0.0

    c_normal = ellipsoid.NormalGravity
    n_normal = min(len(c_normal), len(c1d))

    c1d[:n_normal] -= c_normal[:n_normal]

    cqlm = _triangular_1d_to_lm(c1d, geoid_lmax)
    sqlm = _triangular_1d_to_lm(s1d, geoid_lmax)

    # Old SynthesisType.Geoidheight factor
    geoid_factor = (
            ellipsoid.GM
            / ellipsoid.SemimajorAxis
            / ellipsoid.je
    )

    cqlm = cqlm * geoid_factor
    sqlm = sqlm * geoid_factor

    har = Harmonic(
        lmax=geoid_lmax,
        grid_type=None,
        lat=target_lat,
        lon=target_lon,
    )

    undulation = har.synthesis(cqlm, sqlm)

    return undulation


def load_actual_earth_fields(
        target_lat: np.ndarray,
        target_lon: np.ndarray,
        phisfc_file: str,
        gif48_file: str,
        geoid_lmax: int = 160,
        ellipsoid_type: EllipsoidType = EllipsoidType.GRS80_IERS2010,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load all external fields required by ActualEarth correction.

    Returns
    -------
    orography:
        ERA5 surface geopotential, unit m^2/s^2.
        Shape (nlat, nlon)

    undulation:
        Geoid undulation, unit m.
        Shape (nlat, nlon)
    """

    orography = read_era5_surface_geopotential(
        nc_file=phisfc_file,
        target_lat=target_lat,
        target_lon=target_lon,
        variable="z",
        time_index=0,
    )

    undulation = compute_geoid_undulation_from_gif48(
        gif48_file=gif48_file,
        target_lat=target_lat,
        target_lon=target_lon,
        geoid_lmax=geoid_lmax,
        ellipsoid_type=ellipsoid_type,
    )

    return orography, undulation


if __name__ == "__main__":
    pass
