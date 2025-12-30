import numpy as np

from sagea.constant import PhysicalDimension, GeoConstant


class ConvertSHCConfig:
    def __init__(self):
        self.ln = None

        self.input_field_type = PhysicalDimension.Geopotential
        self.output_field_type = PhysicalDimension.Geopotential

    def set_input_type(self, field_type: PhysicalDimension):
        self.input_field_type = field_type
        return self

    def set_output_type(self, field_type: PhysicalDimension):
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

    def apply_to(self, cs1d):

        length_of_cs1d = np.shape(cs1d)[-1]
        lmax = int(np.sqrt(length_of_cs1d) - 1)

        convert_array = self._get_convert_array_to_dimensionless(
            self.configuration.input_field_type,
            lmax
        ) * self._get_convert_array_from_dimensionless_to(
            self.configuration.output_field_type, lmax
        )
        # [k1, k2, ...,]

        convert_weight_cs1d = np.array([])
        for i in range(lmax + 1):
            convert_weight_cs1d = np.concatenate([convert_weight_cs1d, [convert_array[i]] * (2 * i + 1)])

        cs1d_converted = cs1d * convert_weight_cs1d

        return cs1d_converted

    def _get_convert_array_to_dimensionless(self, field_type: PhysicalDimension, lmax):
        """
        return: [k1, k2, ..., kl, ...]
        """
        assert self.configuration.ln is not None
        ln = self.configuration.ln

        convert_mat = np.ones((lmax + 1,))

        density_water = GeoConstant.density_water
        density_earth = GeoConstant.density_earth
        radius_e = GeoConstant.radius_earth
        GM = GeoConstant.GM
        g_wmo = GeoConstant.g_wmo

        if field_type is PhysicalDimension.Geopotential:
            pass

        elif field_type is PhysicalDimension.EWH:
            ln = ln[:lmax + 1]
            kl = np.array([(1 + ln[n]) / (2 * n + 1) for n in range(len(ln))]) * 3 * density_water / (
                    radius_e * density_earth)

            convert_mat *= kl

        elif field_type is PhysicalDimension.MassDensity:
            ln = ln[:lmax + 1]
            kl = np.array([(1 + ln[n]) / (2 * n + 1) for n in range(len(ln))]) * 3 / (radius_e * density_earth)

            convert_mat *= kl

        elif field_type is PhysicalDimension.Geoid:
            convert_mat /= radius_e

        elif field_type is PhysicalDimension.Gravity:
            kl = np.array([n - 1 for n in range(lmax + 1)]) * (GM / radius_e ** 2)
            convert_mat /= kl

        # elif field_type is PhysicalDimensions.VerticalDisplacement:
        #     lnh, lnl = self.__get_love_number_h_and_l(lmax)
        #
        #     ln = ln[:lmax + 1]
        #     kl = np.array([lnh[n] / (1 + ln[n]) for n in range(len(ln))]) * radius_e
        #
        #     convert_mat /= kl

        elif field_type is PhysicalDimension.Pressure:
            termI = np.arange(lmax + 1)
            term = 2 * termI + 1.
            ln = ln[:lmax + 1]

            kl = (radius_e * density_earth / 3) * (term / (1 + ln)) * g_wmo

            convert_mat /= kl

        else:
            raise Exception

        return convert_mat

    def _get_convert_array_from_dimensionless_to(self, field_type: PhysicalDimension, lmax):
        """
        return: [k1, k2, ..., kl, ...]
        """

        assert self.configuration.ln is not None
        ln = self.configuration.ln

        density_water = GeoConstant.density_water
        density_earth = GeoConstant.density_earth
        radius_e = GeoConstant.radius_earth
        GM = GeoConstant.GM
        g_wmo = GeoConstant.g_wmo

        convert_mat = np.ones((lmax + 1,))

        if field_type is PhysicalDimension.Geopotential:
            pass

        elif field_type is PhysicalDimension.EWH:
            ln = ln[:lmax + 1]
            kl = np.array([(2 * n + 1) / (1 + ln[n]) for n in range(len(ln))]) * radius_e * density_earth / (
                    3 * density_water)

            convert_mat *= kl

        elif field_type is PhysicalDimension.MassDensity:
            ln = ln[:lmax + 1]
            kl = np.array([(2 * n + 1) / (1 + ln[n]) for n in range(len(ln))]) * radius_e * density_earth / 3

            convert_mat *= kl

        elif field_type is PhysicalDimension.Geoid:
            convert_mat *= radius_e

        elif field_type is PhysicalDimension.Gravity:
            kl = np.array([n - 1 for n in range(lmax + 1)]) * (GM / radius_e ** 2)
            convert_mat *= kl

        # elif field_type is PhysicalDimension.VerticalDisplacement:
        #     lnh, lnl = self.__get_love_number_h_and_l(lmax)
        #
        #     ln = ln[:lmax + 1]
        #     kl = np.array([lnh[n] / (1 + ln[n]) for n in range(len(ln))]) * radius_e
        #
        #     convert_mat *= kl

        elif field_type is PhysicalDimension.Pressure:
            termI = np.arange(lmax + 1)
            term = 2 * termI + 1.
            ln = ln[:lmax + 1]

            kl = (radius_e * density_earth / 3) * (term / (1 + ln)) * g_wmo

            convert_mat *= kl

        # elif field_type in (
        #         PhysicalDimensions.HorizontalDisplacementNorth, PhysicalDimensions.HorizontalDisplacementEast):
        #     lnh, lnl = self.__get_love_number_h_and_l(lmax)
        #
        #     ln = ln[:lmax + 1]
        #     kl = np.array([lnl[n] / (1 + ln[n]) for n in range(len(ln))]) * radius_e
        #
        #     convert_mat *= kl

        else:
            raise Exception

        return convert_mat

    # @staticmethod
    # def __get_love_number_h_and_l(lmax):
    #     LN = LoveNumber()
    #
    #     LN.configuration.set_lmax(lmax).set_type(LoveNumberType.VerticalDisplacement)  # h
    #     ln_h = LN.get_Love_number()
    #
    #     LN.configuration.set_lmax(lmax).set_type(LoveNumberType.HorizontalDisplacement)  # l
    #     ln_l = LN.get_Love_number()
    #
    #     return ln_h, ln_l
