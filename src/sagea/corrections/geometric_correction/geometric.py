#!/use/bin/env python
# coding=utf-8
# @Author  : Shuhao Liu
# @Time    : 2026/6/18 17:36 
# @File    : geometric.py

from __future__ import annotations

import numpy as np

from sagea.harmonics.harmonic import Harmonic, GRDType
from sagea.harmonics.transform import _make_harmonic

from .Setting import (
    FieldType,
    Assumption,
    Constants,
    EllipsoidType,
    LoveNumberType,
)
from .RefEllipsoid import RefEllipsoid
from .LoveNumber import LoveNumber


# ============================================================
# Basic helpers
# ============================================================

def _as_3d_cs(
        cqlm: np.ndarray,
        sqlm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Normalize SHC matrices to shape:
        (ntime, lmax + 1, lmax + 1)
    """

    cqlm = np.asarray(cqlm, dtype=float)
    sqlm = np.asarray(sqlm, dtype=float)

    if cqlm.shape != sqlm.shape:
        raise ValueError(
            f"cqlm/sqlm shape mismatch: {cqlm.shape} vs {sqlm.shape}."
        )

    if cqlm.ndim == 2:
        return cqlm[None, :, :], sqlm[None, :, :], True

    if cqlm.ndim == 3:
        return cqlm, sqlm, False

    raise ValueError(
        f"cqlm/sqlm should be 2D or 3D arrays, got {cqlm.shape}."
    )


def _normalize_field_type(field_type) -> FieldType:
    if isinstance(field_type, FieldType):
        return field_type

    if isinstance(field_type, str):
        if field_type in FieldType.__members__:
            return FieldType[field_type]

    raise ValueError(f"Invalid FieldType: {field_type}")


def _normalize_assumption(assumption) -> Assumption:
    if isinstance(assumption, Assumption):
        return assumption

    if isinstance(assumption, str):
        if assumption in Assumption.__members__:
            return Assumption[assumption]

    raise ValueError(f"Invalid Assumption: {assumption}")


def _normalize_love_number_method(method) -> LoveNumberType:
    if isinstance(method, LoveNumberType):
        return method

    if isinstance(method, str):
        if method in LoveNumberType.__members__:
            return LoveNumberType[method]

    raise ValueError(f"Invalid LoveNumberType: {method}")


def _get_love_number_array(
        love_number,
        lmax: int,
        love_number_method: LoveNumberType = LoveNumberType.Wang,
) -> np.ndarray:
    """
    Accept:
    1. ndarray-like love number;
    2. old LoveNumber object;
    """

    if isinstance(love_number, LoveNumber):
        ln = love_number.getNumber(lmax, love_number_method)

    else:
        ln = np.asarray(love_number, dtype=float)

    if ln.ndim != 1:
        raise ValueError(f"love_number should be 1D, got shape {ln.shape}.")

    if len(ln) < lmax + 1:
        raise ValueError(
            f"love_number length should be at least {lmax + 1}, got {len(ln)}."
        )

    return ln[:lmax + 1]


# ============================================================
# Physical factors from old Harmonic.__factorSynthesis
# and old Harmonic._factorHarAnalysis
# ============================================================

def synthesis_factor(
        lmax: int,
        field_type: FieldType,
        love_number: np.ndarray,
        ellipsoid: RefEllipsoid,
) -> np.ndarray:
    """
    Geopotential SHC -> physical grid factor.

    This corresponds to old:
        Harmonic.__factorSynthesis(..., SynthesisType.EWH/Pressure)

    Parameters
    ----------
    field_type:
        FieldType.EWH or FieldType.Pressure

    Returns
    -------
    factor : ndarray, shape (lmax + 1,)
        Degree-wise factor.
    """

    l = np.arange(lmax + 1, dtype=float)
    term = 2.0 * l + 1.0

    R = ellipsoid.SemimajorAxis
    rho_ave = ellipsoid.rho_ave
    kl = love_number[:lmax + 1]

    if field_type == FieldType.EWH:
        return (
                R * rho_ave / 3.0
                * term / (1.0 + kl)
                / Constants.rho_water
        )

    if field_type == FieldType.Pressure:
        return (
                R * rho_ave / 3.0
                * term / (1.0 + kl)
                * Constants.g_wmo
        )

    raise ValueError(f"Unsupported field_type: {field_type}")


def inner_integral_analysis_factor(
        lmax: int,
        field_type: FieldType,
        love_number: np.ndarray,
        ellipsoid: RefEllipsoid,
) -> np.ndarray:
    """
    Physical inner integral -> geopotential SHC factor.

    This corresponds to old:
        HarAnalysisType.InnerIntegral
        HarAnalysisType.InnerIntegral_EWH
    """

    l = np.arange(lmax + 1, dtype=float)
    term = 2.0 * l + 1.0

    R = ellipsoid.SemimajorAxis
    rho_ave = ellipsoid.rho_ave
    kl = love_number[:lmax + 1]

    if field_type == FieldType.Pressure:
        # old HarAnalysisType.InnerIntegral
        return 1.0 / (
                R * rho_ave / 3.0
                * term / (1.0 + kl)
        )

    if field_type == FieldType.EWH:
        # old HarAnalysisType.InnerIntegral_EWH
        return 1.0 / (
                R * rho_ave / 3.0
                * term / (1.0 + kl)
                / Constants.rho_water
        )

    raise ValueError(f"Unsupported field_type: {field_type}")


# ============================================================
# Geometry formulas copied from old ForwardModel
# ============================================================

def ellipsoid_radius_at_lat(
        lat_deg: np.ndarray,
        ellipsoid: RefEllipsoid,
) -> np.ndarray:
    """
    Old ForwardModel._getR(lat)
    """

    B = np.deg2rad(lat_deg)

    R = ellipsoid.SemimajorAxis
    e = ellipsoid.Eccentricity

    return R * np.sqrt(1.0 - e ** 2 * np.sin(B) ** 2)


def real_height_from_geopotential_height(
        lat_deg: np.ndarray,
        H: np.ndarray,
) -> np.ndarray:
    """
    Old ForwardModel._getHeight(theta, H)

    Parameters
    ----------
    lat_deg:
        Latitude in degrees.

    H:
        Geopotential height.
        In old code:
            H = orography / Constants.g_wmo
    """

    B = np.deg2rad(90.0 - lat_deg)

    z = (
            (1.0 - 0.002644 * np.cos(2.0 * B)) * H
            + (1.0 - 0.0089 * np.cos(2.0 * B))
            * H ** 2
            / 6.245e6
    )

    return z


def gravity_at_height(
        lat_deg: np.ndarray,
        z: np.ndarray,
        ellipsoid: RefEllipsoid,
) -> np.ndarray:
    """
    Old ForwardModel._getG(theta, z)
    """

    B = np.deg2rad(90.0 - lat_deg)

    R = ellipsoid.SemimajorAxis

    g = (
            ellipsoid.je
            * (
                    1.0
                    + 5.2885e-3 * np.cos(B) ** 2
                    - 5.9e-6 * np.cos(2.0 * B) ** 2
            )
            * (
                    1.0
                    - 2.0
                    * (
                            1.006803
                            - 0.060706 * np.cos(B) ** 2
                    )
                    * z / R
                    + 3.0 * (z / R) ** 2
            )
    )

    return g


# ============================================================
# Forward model: physical grid -> geopotential SHC
# ============================================================

class GeometricForwardModel:
    """
    Vectorized replacement of old ForwardModel.getCS().

    Important
    ---------
    This class does NOT know SHC class.
    It only accepts numpy grids and returns cqlm/sqlm.
    """

    def __init__(
            self,
            lmax: int,
            lat: np.ndarray,
            lon: np.ndarray,
            harmonic: Harmonic,
            love_number: np.ndarray,
            assumption: Assumption = Assumption.ActualEarth,
            field_type: FieldType = FieldType.EWH,
            ellipsoid: RefEllipsoid | None = None,
            orography: np.ndarray | None = None,
            undulation: np.ndarray | None = None,
    ):
        self.lmax = int(lmax)

        self.lat = np.asarray(lat, dtype=float)
        self.lon = np.asarray(lon, dtype=float)

        self.harmonic = harmonic

        self.assumption = _normalize_assumption(assumption)
        self.field_type = _normalize_field_type(field_type)

        self.ellipsoid = (
            RefEllipsoid(EllipsoidType.GRS80_IERS2010)
            if ellipsoid is None
            else ellipsoid
        )

        self.love_number = np.asarray(love_number, dtype=float)

        self.analysis_factor = inner_integral_analysis_factor(
            lmax=self.lmax,
            field_type=self.field_type,
            love_number=self.love_number,
            ellipsoid=self.ellipsoid,
        )

        self._prepare_geometry(
            orography=orography,
            undulation=undulation,
        )

    def _prepare_geometry(
            self,
            orography: np.ndarray | None,
            undulation: np.ndarray | None,
    ):
        nlat = len(self.lat)
        nlon = len(self.lon)

        lat2d = np.broadcast_to(
            self.lat[:, None],
            (nlat, nlon),
        )

        if orography is None:
            orography = np.zeros((nlat, nlon), dtype=float)
        else:
            orography = np.asarray(orography, dtype=float).reshape(nlat, nlon)

        if undulation is None:
            undulation = np.zeros((nlat, nlon), dtype=float)
        else:
            undulation = np.asarray(undulation, dtype=float).reshape(nlat, nlon)

        self.orography = orography
        self.undulation = undulation

        # old:
        # z = self._getHeight(latMesh, self.__orography / Constants.g_wmo)
        z = real_height_from_geopotential_height(
            lat_deg=lat2d,
            H=orography / Constants.g_wmo,
        )

        self.height = z

        if self.assumption == Assumption.ActualEarth:
            r = (
                    ellipsoid_radius_at_lat(lat2d, self.ellipsoid)
                    + undulation
                    + z
            )

        elif self.assumption == Assumption.Ellipsoid:
            r = ellipsoid_radius_at_lat(lat2d, self.ellipsoid)

        elif self.assumption == Assumption.Sphere:
            r = np.zeros_like(lat2d)

        else:
            raise ValueError(f"Unsupported assumption: {self.assumption}")

        self.r = r

        # old:
        # ar = r / ellipsoid.SemimajorAxis
        self.ar = r / self.ellipsoid.SemimajorAxis

        # old:
        # gr = self._getG(latMesh, z)
        self.gravity = gravity_at_height(
            lat_deg=lat2d,
            z=z,
            ellipsoid=self.ellipsoid,
        )

    def field_to_shc(
            self,
            field: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert EWH/Pressure grid to geopotential SHC.

        Parameters
        ----------
        field:
            Shape:
                - (nlat, nlon)
                - (ntime, nlat, nlon)

        Returns
        -------
        cqlm, sqlm:
            Shape:
                - (ntime, lmax + 1, lmax + 1)
        """

        field = np.asarray(field, dtype=float)

        if field.ndim == 2:
            field = field[None, :, :]

        if field.ndim != 3:
            raise ValueError(
                f"field should be 2D or 3D, got shape {field.shape}."
            )

        ntime, nlat, nlon = field.shape

        if (nlat, nlon) != (len(self.lat), len(self.lon)):
            raise ValueError(
                f"field shape mismatch. Expected "
                f"{(len(self.lat), len(self.lon))}, got {(nlat, nlon)}."
            )

        # old:
        # if kind == FieldType.Pressure:
        #     deltaI = field / gr
        # else:
        #     deltaI = field
        if self.field_type == FieldType.Pressure:
            base = field / self.gravity[None, :, :]
        elif self.field_type == FieldType.EWH:
            base = field
        else:
            raise ValueError(f"Unsupported field_type: {self.field_type}")

        cqlm = np.zeros(
            (ntime, self.lmax + 1, self.lmax + 1),
            dtype=float,
        )
        sqlm = np.zeros_like(cqlm)

        # ------------------------------------------------------------
        # Sphere case
        # old:
        #     deltaI = [field] or [field/gr]
        #     hm.analysis(..., Inner=[...])
        # ------------------------------------------------------------
        if self.assumption == Assumption.Sphere:
            c_tmp, s_tmp = self.harmonic.analysis(base)

            cqlm = c_tmp * self.analysis_factor[None, :, None]
            sqlm = s_tmp * self.analysis_factor[None, :, None]

            return cqlm, sqlm

        # ------------------------------------------------------------
        # Non-sphere case
        #
        # old:
        #     iniPower = ar
        #     for i in range(maxDeg + 1):
        #         iniPower = iniPower * ar
        #         I_lev = iniPower * field
        #
        # Therefore:
        #     degree l uses ar ** (l + 2) * field
        # ------------------------------------------------------------
        ar_power = self.ar[None, :, :] ** 2

        for l in range(self.lmax + 1):
            field_l = base * ar_power

            clm_l, slm_l = self.harmonic.analysis_degree(
                field_l,
                degree=l,
            )

            cqlm[:, l, :l + 1] = (
                    clm_l[:, :l + 1]
                    * self.analysis_factor[l]
            )
            sqlm[:, l, :l + 1] = (
                    slm_l[:, :l + 1]
                    * self.analysis_factor[l]
            )

            ar_power = ar_power * self.ar[None, :, :]

        return cqlm, sqlm


# ============================================================
# Main correction function
# ============================================================

def apply_geometric_correction(
        cqlm: np.ndarray,
        sqlm: np.ndarray,
        love_number,
        assumption="ActualEarth",
        field_type="EWH",
        love_number_method="Wang",
        ellipsoid=None,
        grid_type=None,
        grid_space: float | None = 0.5,
        orography: np.ndarray | None = None,
        undulation: np.ndarray | None = None,
        phisfc_file: str | None = None,
        gif48_file: str | None = None,
        geoid_lmax: int = 160,
        auto_load_actual_earth: bool = False,
        iter_max: int = 4,
        vmax: float = 2.5,
        vmin: float = 0.0,
        eps: float = 0.0,
        log: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized geometrical correction.

    Parameters
    ----------
    cqlm, sqlm:
        Geopotential spherical harmonic coefficients.
        Shape:
            - (lmax + 1, lmax + 1)
            - (ntime, lmax + 1, lmax + 1)

    love_number:
        ndarray or old LoveNumber object.

    assumption:
        "Sphere", "Ellipsoid", "ActualEarth"

    field_type:
        "EWH" or "Pressure"

    grid_space:
        If given, use custom regular geographic grid.

    grid_type:
        If grid_space is None, use GLQ/DH/DH2 grid.

    orography:
        Surface geopotential, same meaning as old PHISFC.
        Shape should be (nlat, nlon).

    undulation:
        Geoid undulation.
        Shape should be (nlat, nlon).

    Returns
    -------
    cqlm_corr, sqlm_corr
        Same dimension type as input.
    """

    cqlm0, sqlm0, single = _as_3d_cs(cqlm, sqlm)

    lmax = cqlm0.shape[1] - 1

    assumption = _normalize_assumption(assumption)
    field_type = _normalize_field_type(field_type)
    love_number_method = _normalize_love_number_method(love_number_method)

    ellipsoid = (
        RefEllipsoid(EllipsoidType.GRS80_IERS2010)
        if ellipsoid is None
        else ellipsoid
    )

    ln = _get_love_number_array(
        love_number=love_number,
        lmax=lmax,
        love_number_method=love_number_method,
    )

    # ------------------------------------------------------------
    # Prepare harmonic operator and grid coordinates.
    # ------------------------------------------------------------
    harmonic, lat, lon = _make_harmonic(
        lmax=lmax,
        grid_type=grid_type,
        grid_space=grid_space,
    )

    # ------------------------------------------------------------
    # ActualEarth requires external geometry fields.
    # ------------------------------------------------------------
    if assumption == Assumption.ActualEarth:
        missing_orography = orography is None
        missing_undulation = undulation is None

        if missing_orography or missing_undulation:

            if not auto_load_actual_earth:
                raise ValueError(
                    "assumption='ActualEarth' requires external fields: "
                    "orography and undulation. "
                    "Please provide them explicitly, or set "
                    "auto_load_actual_earth=True with phisfc_file and gif48_file."
                )

            if phisfc_file is None:
                raise ValueError(
                    "phisfc_file is required when auto_load_actual_earth=True "
                    "and orography is None."
                )

            if gif48_file is None:
                raise ValueError(
                    "gif48_file is required when auto_load_actual_earth=True "
                    "and undulation is None."
                )

            from sagea.corrections.geometric_correction.external_fields import (
                load_actual_earth_fields,
            )

            orography_loaded, undulation_loaded = load_actual_earth_fields(
                target_lat=lat,
                target_lon=lon,
                phisfc_file=phisfc_file,
                gif48_file=gif48_file,
                geoid_lmax=geoid_lmax,
                ellipsoid_type=EllipsoidType.GRS80_IERS2010,
            )

            if orography is None:
                orography = orography_loaded

            if undulation is None:
                undulation = undulation_loaded

    # ------------------------------------------------------------
    # Degree-wise synthesis factor:
    # geopotential SHC -> EWH/Pressure grid
    # ------------------------------------------------------------
    syn_factor = synthesis_factor(
        lmax=lmax,
        field_type=field_type,
        love_number=ln,
        ellipsoid=ellipsoid,
    )

    # ------------------------------------------------------------
    # Forward model:
    # EWH/Pressure grid -> geopotential SHC under selected geometry
    # ------------------------------------------------------------
    forward_model = GeometricForwardModel(
        lmax=lmax,
        lat=lat,
        lon=lon,
        harmonic=harmonic,
        love_number=ln,
        assumption=assumption,
        field_type=field_type,
        ellipsoid=ellipsoid,
        orography=orography,
        undulation=undulation,
    )

    # ------------------------------------------------------------
    # Old code:
    #
    # CnmT0, SnmT0 = input
    # CnmT, SnmT = CnmT0, SnmT0
    # gg = synthesis(CnmT, SnmT)
    #
    # for iter:
    #     CnmR, SnmR = fm.getCS(field=gg)
    #     xr = CnmT / CnmR
    #     CnmT = xr * CnmT0
    #     gg = synthesis(CnmT)
    # ------------------------------------------------------------
    cqlm_t = cqlm0.copy()
    sqlm_t = sqlm0.copy()

    # initial physical field from true CS under spherical synthesis
    grid = harmonic.synthesis(
        cqlm_t * syn_factor[None, :, None],
        sqlm_t * syn_factor[None, :, None],
    )

    for it in range(iter_max):
        if log:
            print(f"geometric correction iteration {it + 1}/{iter_max}")

        cqlm_r, sqlm_r = forward_model.field_to_shc(grid)

        ratio_c = np.ones_like(cqlm_t)
        ratio_s = np.ones_like(sqlm_t)

        if eps <= 0:
            valid_c = cqlm_r != 0
            valid_s = sqlm_r != 0
        else:
            valid_c = np.abs(cqlm_r) > eps
            valid_s = np.abs(sqlm_r) > eps

        ratio_c[valid_c] = cqlm_t[valid_c] / cqlm_r[valid_c]
        ratio_s[valid_s] = sqlm_t[valid_s] / sqlm_r[valid_s]

        # old:
        # xr_C[np.isnan(xr_C)] = 1
        # xr_S[np.isnan(xr_S)] = 1
        ratio_c[~np.isfinite(ratio_c)] = 1.0
        ratio_s[~np.isfinite(ratio_s)] = 1.0

        # old:
        # xr_C[np.fabs(xr_C) > Vmax] = 1
        # xr_C[np.fabs(xr_C) < Vmin] = 1
        ratio_c[np.abs(ratio_c) > vmax] = 1.0
        ratio_s[np.abs(ratio_s) > vmax] = 1.0

        ratio_c[np.abs(ratio_c) < vmin] = 1.0
        ratio_s[np.abs(ratio_s) < vmin] = 1.0

        # old:
        # CnmT = xr_C * CnmT0
        # SnmT = xr_S * SnmT0
        cqlm_t = ratio_c * cqlm0
        sqlm_t = ratio_s * sqlm0

        cqlm_t[~np.isfinite(cqlm_t)] = 0.0
        sqlm_t[~np.isfinite(sqlm_t)] = 0.0

        # update grid
        grid = harmonic.synthesis(
            cqlm_t * syn_factor[None, :, None],
            sqlm_t * syn_factor[None, :, None],
        )

    if single:
        return cqlm_t[0], sqlm_t[0]

    return cqlm_t, sqlm_t


if __name__ == "__main__":
    pass
