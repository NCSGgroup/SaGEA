import numpy as np

from pysrc.auxiliary.core.SHC import SHC
from pysrc.auxiliary.preference.EnumClasses import FieldPhysicalQuantity, LoveNumberType
from pysrc.auxiliary.preference.Constants import GeoConstants
from pysrc.auxiliary.tools.MathTool import MathTool
from pysrc.post_processing.Love_number.LoveNumber import LoveNumber


class ConvertSHCConfig:
    def __init__(self):
        self.ln = None

        self.input_field_type = FieldPhysicalQuantity.Dimensionless
        self.output_field_type = FieldPhysicalQuantity.EWH

    def set_input_type(self, field_type: FieldPhysicalQuantity):
        self.input_field_type = field_type
        return self

    def set_output_type(self, field_type: FieldPhysicalQuantity):
        self.output_field_type = field_type
        return self

    def set_Love_number(self, ln):
        self.ln = ln
        return self


class ConvertSHC:
    def __init__(self):
        self.configuration = ConvertSHCConfig()

    def config(self, config: ConvertSHCConfig):
        self.configuration = config
        return self

    def apply_to(self, shc: SHC):

        lmax = shc.get_lmax()

        convert_array = self._get_convert_array_to_dimensionless(lmax) * self._get_convert_array_from_dimensionless_to(
            self.configuration.output_field_type, lmax)
        # [k1, k2, ...,]

        convert_weight_cs1d = np.array([])
        for i in range(lmax + 1):
            convert_weight_cs1d = np.concatenate([convert_weight_cs1d, [convert_array[i]] * (2 * i + 1)])

        cs1d_converted = shc.cs * convert_weight_cs1d

        return SHC(cs1d_converted)

    def _get_convert_array_to_dimensionless(self, lmax):
        """
        return: [k1, k2, ..., kl, ...]
        """
        assert self.configuration.ln is not None
        ln = self.configuration.ln

        convert_mat = np.ones((lmax + 1,))

        density_water = GeoConstants.density_water
        density_earth = GeoConstants.density_earth
        radius_e = GeoConstants.radius_earth

        if self.configuration.input_field_type is FieldPhysicalQuantity.Dimensionless:
            pass

        elif self.configuration.input_field_type is FieldPhysicalQuantity.EWH:
            ln = ln[:lmax + 1]
            kl = np.array([(1 + ln[n]) / (2 * n + 1) for n in range(len(ln))]) * 3 * density_water / (
                    radius_e * density_earth)

            convert_mat *= kl

        elif self.configuration.input_field_type is FieldPhysicalQuantity.Density:
            ln = ln[:lmax + 1]
            kl = np.array([(1 + ln[n]) / (2 * n + 1) for n in range(len(ln))]) * 3 / (radius_e * density_earth)

            convert_mat *= kl

        else:
            raise Exception

        return convert_mat

    def _get_convert_array_from_dimensionless_to(self, field_type: FieldPhysicalQuantity, lmax):
        """
        return: [k1, k2, ..., kl, ...]
        """

        assert self.configuration.ln is not None
        ln = self.configuration.ln

        def _get_love_number_h_and_l():
            LN = LoveNumber()

            LN.configuration.set_lmax(lmax).set_type(LoveNumberType.VerticalDisplacement)  # h
            ln_h = LN.get_Love_number()

            LN.configuration.set_lmax(lmax).set_type(LoveNumberType.HorizontalDisplacement)  # l
            ln_l = LN.get_Love_number()

            return ln_h, ln_l

        density_water = GeoConstants.density_water
        density_earth = GeoConstants.density_earth
        radius_e = GeoConstants.radius_earth
        GM = GeoConstants.GM

        convert_mat = np.ones((lmax + 1,))

        if field_type is FieldPhysicalQuantity.Dimensionless:
            return self

        elif field_type is FieldPhysicalQuantity.EWH:
            ln = ln[:lmax + 1]
            kl = np.array([(2 * n + 1) / (1 + ln[n]) for n in range(len(ln))]) * radius_e * density_earth / (
                    3 * density_water)

            convert_mat *= kl

        elif field_type is FieldPhysicalQuantity.Density:
            ln = ln[:lmax + 1]
            kl = np.array([(2 * n + 1) / (1 + ln[n]) for n in range(len(ln))]) * radius_e * density_earth / 3

            convert_mat *= kl

        elif field_type is FieldPhysicalQuantity.Geoid:
            convert_mat *= radius_e

        elif field_type is FieldPhysicalQuantity.Gravity:
            kl = np.array([n - 1 for n in range(lmax + 1)]) * (GM / radius_e ** 2)
            convert_mat *= kl

        elif field_type is FieldPhysicalQuantity.VerticalDisplacement:
            lnh, lnl = _get_love_number_h_and_l()

            ln = ln[:lmax + 1]
            kl = np.array([lnh[n] / (1 + ln[n]) for n in range(len(ln))]) * radius_e

            convert_mat *= kl

        elif field_type in (
                FieldPhysicalQuantity.HorizontalDisplacementNorth, FieldPhysicalQuantity.HorizontalDisplacementEast):
            lnh, lnl = _get_love_number_h_and_l()

            ln = ln[:lmax + 1]
            kl = np.array([lnl[n] / (1 + ln[n]) for n in range(len(ln))]) * radius_e

            convert_mat *= kl

        else:
            raise Exception

        return convert_mat
