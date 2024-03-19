"""
@Company: CGE-HUST, Wuhan, China
@Author: Yang Fan
@Contact: yfan_cge@hust.edu.cn
@Modify Time:2020/11/2 上午11:12
@Description:
"""

import numpy as np
from pysrc.post_processing.geometric_correction.original_files.GeoMathKit import GeoMathKit


class Gaussion:

    def __init__(self):
        self.__a = 6.3781363000E+06  # radius of the earth
        self.__w = None
        self.__Cnm = None
        self.__Snm = None

        pass

    def setCS(self, Cnm, Snm):
        Cnm = GeoMathKit.CS_1dTo2d(Cnm)
        Snm = GeoMathKit.CS_1dTo2d(Snm)

        self.__Cnm = Cnm
        self.__Snm = Snm
        return self

    def setRadius(self, radius, Nmax):
        """
        :param Nmax: Max degree of SHs to be filtered
        :param radius: half-width radius of Gauss filter, unit [km]
        :return:
        """
        assert Nmax >= 1

        '''
        convert [km] to [m]
        '''
        radius = radius * 1000

        b = np.log(2.) / (1 - np.cos(radius / self.__a))
        w = np.zeros(Nmax + 1)
        w[0] = 1. / (2 * np.pi)
        w[1] = 1. / (2 * np.pi) * ((1 + np.exp(-2 * b)) / (1 - np.exp(-2 * b)) - 1. / b)

        for l in range(1, Nmax):
            w[l + 1] = -(2 * l + 1) / b * w[l] + w[l - 1]

        '''
        all multiplied with 2*pi to ease the computations 
        '''

        self.__w = w * 2. * np.pi

        return self

    def getCS(self):
        """

        :return: 1-d CS that has been filtered with specified Gauss weights
        """
        sh = np.shape(self.__Cnm)
        assert len(self.__w) >= sh[0]

        for i in range(sh[0]):
            self.__Cnm[i,] = self.__Cnm[i,] * self.__w[i]
            self.__Snm[i,] = self.__Snm[i,] * self.__w[i]

        Cnm = GeoMathKit.CS_2dTo1d(self.__Cnm)
        Snm = GeoMathKit.CS_2dTo1d(self.__Snm)

        return Cnm, Snm

