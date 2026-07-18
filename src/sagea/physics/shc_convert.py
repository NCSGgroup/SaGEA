from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sagea.constants.constant import PhysicalDimension, GeoConstant


@dataclass
class ConvertSHCConfig:
    ln: np.ndarray | None = None
    input_field_type: PhysicalDimension = PhysicalDimension.Geopotential
    output_field_type: PhysicalDimension = PhysicalDimension.Geopotential

    def set_input_type(self, field_type: PhysicalDimension):
        self.input_field_type = field_type
        return self

    def set_output_type(self, field_type: PhysicalDimension):
        self.output_field_type = field_type
        return self

    def set_Love_number(self, ln: np.ndarray):
        self.ln = np.asarray(ln, dtype=float)
        return self


class ConvertSHC:
    """
    Convert spherical harmonic coefficients between physical dimensions.

    Input cs1d shape:
    - (ncoef,)
    - (ntime, ncoef)

    Output shape is same as input.
    """

    def __init__(self):
        self.configuration = ConvertSHCConfig()

    def config(self, config: ConvertSHCConfig):
        self.configuration = config
        return self

    def apply_to(self, cs1d: np.ndarray) -> np.ndarray:
        cs1d = np.asarray(cs1d, dtype=float)

        if cs1d.ndim not in (1, 2):
            raise ValueError("cs1d should be a 1D or 2D array.")

        ncoef = cs1d.shape[-1]
        lmax_approx = np.sqrt(ncoef) - 1

        if not np.isclose(lmax_approx, round(lmax_approx), atol=1e-8):
            raise ValueError("Invalid cs1d length. Expected (lmax + 1)^2.")

        lmax = int(round(lmax_approx))

        degree_weight = (
            self._get_convert_array_to_dimensionless(
                self.configuration.input_field_type,
                lmax,
            )
            * self._get_convert_array_from_dimensionless_to(
                self.configuration.output_field_type,
                lmax,
            )
        )

        cs_weight = np.repeat(degree_weight, 2 * np.arange(lmax + 1) + 1)

        return cs1d * cs_weight

    def _get_convert_array_to_dimensionless(
        self,
        field_type: PhysicalDimension,
        lmax: int,
    ) -> np.ndarray:
        """
        Convert from physical dimension to dimensionless geopotential coefficients.
        """

        ln = self.__get_love_number(lmax)

        convert = np.ones(lmax + 1)

        rho_w = GeoConstant.density_water
        rho_e = GeoConstant.density_earth
        r_e = GeoConstant.radius_earth
        gm = GeoConstant.GM
        g_wmo = GeoConstant.g_wmo

        degree = np.arange(lmax + 1)

        if field_type is PhysicalDimension.Geopotential:
            pass

        elif field_type is PhysicalDimension.EWH:
            factor = ((1 + ln) / (2 * degree + 1)) * 3 * rho_w / (r_e * rho_e)
            convert *= factor

        elif field_type is PhysicalDimension.MassDensity:
            factor = ((1 + ln) / (2 * degree + 1)) * 3 / (r_e * rho_e)
            convert *= factor

        elif field_type is PhysicalDimension.Geoid:
            convert /= r_e

        elif field_type is PhysicalDimension.Gravity:
            factor = (degree - 1) * (gm / r_e**2)
            convert /= factor

        elif field_type is PhysicalDimension.Pressure:
            factor = (r_e * rho_e / 3) * ((2 * degree + 1) / (1 + ln)) * g_wmo
            convert /= factor

        else:
            raise ValueError(f"Unsupported input physical dimension: {field_type}")

        return convert

    def _get_convert_array_from_dimensionless_to(
        self,
        field_type: PhysicalDimension,
        lmax: int,
    ) -> np.ndarray:
        """
        Convert from dimensionless geopotential coefficients to target physical dimension.
        """

        ln = self.__get_love_number(lmax)

        convert = np.ones(lmax + 1)

        rho_w = GeoConstant.density_water
        rho_e = GeoConstant.density_earth
        r_e = GeoConstant.radius_earth
        gm = GeoConstant.GM
        g_wmo = GeoConstant.g_wmo

        degree = np.arange(lmax + 1)

        if field_type is PhysicalDimension.Geopotential:
            pass

        elif field_type is PhysicalDimension.EWH:
            factor = ((2 * degree + 1) / (1 + ln)) * r_e * rho_e / (3 * rho_w)
            convert *= factor

        elif field_type is PhysicalDimension.MassDensity:
            factor = ((2 * degree + 1) / (1 + ln)) * r_e * rho_e / 3
            convert *= factor

        elif field_type is PhysicalDimension.Geoid:
            convert *= r_e

        elif field_type is PhysicalDimension.Gravity:
            factor = (degree - 1) * (gm / r_e**2)
            convert *= factor

        elif field_type is PhysicalDimension.Pressure:
            factor = (r_e * rho_e / 3) * ((2 * degree + 1) / (1 + ln)) * g_wmo
            convert *= factor

        else:
            raise ValueError(f"Unsupported output physical dimension: {field_type}")

        return convert

    def __get_love_number(self, lmax: int) -> np.ndarray:
        if self.configuration.ln is None:
            raise ValueError("Love number is required for physical conversion.")

        ln = np.asarray(self.configuration.ln, dtype=float)

        if len(ln) < lmax + 1:
            raise ValueError(
                f"Love number length is insufficient. "
                f"Need {lmax + 1}, got {len(ln)}."
            )

        return ln[: lmax + 1]