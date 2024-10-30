import numpy as np

from pysrc.auxiliary.preference.Constants import GeoConstants

from pysrc.auxiliary.aux_tool.MathTool import MathTool
from pysrc.post_processing.filter.Base import get_gaussian_weight_1d, SHCFilter


class GaussianConfig:
    def __init__(self):
        """default setting"""
        self.filtering_radius = 300 * 1000  # unit[m]
        self.earth_radius = GeoConstants.radius_earth
        self.lmax = 60

    def set_filtering_radius(self, radius: int):
        """
        filtering radius,
        the distance from the center of the kernel to the point where the filtering kernel decays to its half.
        :param radius: int, unit [km]
        :return:
        """
        self.filtering_radius = radius * 1000
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

        info = f'Filtering Radius:\t{self.filtering_radius / 1000} km\n' \
               f'Earth Radius:\t{self.earth_radius / 1000} km\n' \
               f'Max degree/order:\t{self.lmax}'

        return info


class Gaussian(SHCFilter):
    def __init__(self):
        self.configuration = GaussianConfig()

    def config(self, config: GaussianConfig):
        self.configuration = config

    def __get_weight_matrix(self):
        """
        get C/S filtering weight sorted by degree, that is,
        [[w00,   0,   0, ...]
         [w10, w11,   0, ...]
         [w20, w21, w22, ...]
         [       ...       ]],
        for Gaussian weight which is dependent only by degree l, the result is
        [[w0,   0,   0, ...]
         [w1,  w1,   0, ...]
         [w2,  w2,  w2, ...]
         [       ...       ]],
        :return:
        """
        w = get_gaussian_weight_1d(self.configuration.lmax, self.configuration.filtering_radius,
                                   self.configuration.earth_radius)
        wlm = np.tile(w, (self.configuration.lmax + 1, 1)).T

        return np.tril(wlm)

    def get_weight_cs1d(self):
        """
        get Gaussian weight sorted by degree, that is,
        [w(s00), w(c10), w(c11), w(s11), w(c20), w(c21), w(s21), w(c22), w(s22), ...],
        for Gaussian weight which is dependent only by degree l, the result is
        [w0; w1, w1, w1; w2, w2, w2, w2, w2; ...],
        :return: 1-d array in length (self.lmax + 1) **2
        """
        # weight_mat = self._get_weight_matrix()
        # weight1d_half_part = MathTool.cs_2dto1d(weight_mat, sort=MathTool.CS1dSortedBy.order)
        # gs_weight_array = get_gaussian_weight_1d(self.configuration.lmax, self.configuration.filtering_radius,
        #                                          self.configuration.earth_radius)
        # gs_weight_cs1d = np.array([])
        # for i in range(self.configuration.lmax + 1):
        #     gs_weight_cs1d = np.concatenate([gs_weight_cs1d, [gs_weight_array[i]] * (2 * i + 1)])

        weight_mat = self.__get_weight_matrix()
        gs_weight_cs1d = MathTool.cs_combine_to_triangle_1d(weight_mat, weight_mat)

        return gs_weight_cs1d

    def apply_to(self, cqlm, sqlm):
        if self.configuration.filtering_radius == 0:
            return cqlm, sqlm

        cqlm, sqlm, single = self._cs_to_3d_array(cqlm, sqlm)

        weight_matrix = self.__get_weight_matrix()
        cqlm_f, sqlm_f = cqlm * weight_matrix, sqlm * weight_matrix

        if single:
            assert cqlm_f.shape[0] == sqlm_f.shape[0] == 1
            return cqlm_f[0], sqlm_f[0]
        else:
            return cqlm_f, sqlm_f
