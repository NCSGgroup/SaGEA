import numpy as np

from pysrc.auxiliary.preference.Constants import GeoConstants

from pysrc.auxiliary.aux_tool.MathTool import MathTool
from pysrc.post_processing.filter.Base import get_gaussian_weight_1d, SHCFilter


class FanConfig:
    def __init__(self):
        """default setting"""
        self.filtering_radius1 = 500 * 1000  # unit[m]
        self.filtering_radius2 = 500 * 1000  # unit[m]

        self.earth_radius = GeoConstants.radius_earth
        self.lmax = 60

    def set_filtering_params(self, radius1: int, radius2: int):
        """
        filtering radius,
        the distance from the center of the kernel to the point where the filtering kernel decays to its half.
        :param radius1: int, unit [km]
        :param radius2: int, unit [km]
        :return:
        """
        self.filtering_radius1 = radius1 * 1000
        self.filtering_radius2 = radius2 * 1000
        return self

    def set_lmax(self, lmax: int):
        """
        max degree/order
        :param lmax: int
        :return:
        """
        self.lmax = lmax
        return self

    def __str__(self):
        """

        :return: str, information of the configuration
        """

        info = f'Filtering Radius 1:\t{self.filtering_radius1 / 1000} km\n' \
               f'Filtering Radius 2:\t{self.filtering_radius2 / 1000} km\n' \
               f'Earth Radius:\t{self.earth_radius / 1000} km\n' \
               f'Max degree/order:\t{self.lmax}'

        return info


class Fan(SHCFilter):
    def __init__(self):
        self.configuration = FanConfig()

    def config(self, config: FanConfig):
        self.configuration = config

    def __get_weight_matrix(self):
        """
        get C/S filtering weight sorted by degree, that is,
        [[w00,   0,   0, ...]
         [w10, w11,   0, ...]
         [w20, w21, w22, ...]
         [       ...       ]],
        :return:
        """
        lmax = self.configuration.lmax

        w1 = get_gaussian_weight_1d(lmax, self.configuration.filtering_radius1, self.configuration.earth_radius)
        w2 = get_gaussian_weight_1d(lmax, self.configuration.filtering_radius2, self.configuration.earth_radius)

        matrix = w1[:, None] @ w2[None, :]

        return np.tril(matrix)

    def get_weight_cs1d(self):
        """
        get C/S filtering weight sorted by degree, that is,
        [w(s00), w(c10), w(c11), w(s11), w(c20), w(c21), w(s21), w(c22), w(s22), ...]
        :return: 1-d array in length (self.lmax + 1) **2
        """

        weight_mat = self.__get_weight_matrix()
        gs_weight_cs1d = MathTool.cs_combine_to_triangle_1d(weight_mat, weight_mat)

        return gs_weight_cs1d

    def apply_to(self, cqlm, sqlm):

        cqlm, sqlm, single = self._cs_to_3d_array(cqlm, sqlm)

        weight_matrix = self.__get_weight_matrix()
        cqlm_f, sqlm_f = cqlm * weight_matrix, sqlm * weight_matrix

        if single:
            assert cqlm_f.shape[0] == sqlm_f.shape[0] == 1
            return cqlm_f[0], sqlm_f[0]
        else:
            return cqlm_f, sqlm_f
