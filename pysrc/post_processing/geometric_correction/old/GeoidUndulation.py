#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:Yang Fan
# mailbox: yfan_cge@hust.edu.cn
# address: Huazhong University of Science and Technology, Wuhan, China
# datetime:2020/6/15 下午4:41
# software: Atmosphere de-aliasing modelling
# usage of this file: calculate the geoid undulation

import numpy as np

from pysrc.auxiliary.aux_tool.FileTool import FileTool
from pysrc.post_processing.geometric_correction.old.Harmonic import Harmonic
from pysrc.post_processing.geometric_correction.old.LoadSH import LoadSH, Gif48
from pysrc.post_processing.geometric_correction.old.LoveNumber import LoveNumber
from pysrc.post_processing.geometric_correction.old.RefEllipsoid import EllipsoidType, RefEllipsoid
from pysrc.post_processing.geometric_correction.old.Setting import SynthesisType, LoveNumberType


class GeoidUndulation:
    """
    Definitions and calculations refer to the
    http://icgem.gfz-potsdam.de/faq

    Notice: A more detailed and more accurate computation refer to:
    https://www.sciencedirect.com/science/article/pii/B9780444527486001565
    https://shtools.github.io/SHTOOLS/pymakegeoidgriddh.html
    """

    __LN = LoveNumber('../data/')

    def __init__(self, ell: EllipsoidType):
        self.__SH = None
        self.__Nmax = None

        self.__default()

        self.__ellipsoid = RefEllipsoid(ell)
        self.__Har = Harmonic(GeoidUndulation.__LN, Parallel=-1).setEllipsoid(ell).setLoveNumMethod(LoveNumberType.Wang)

        pass

    def setGravityModel(self, SH: LoadSH, Nmax: int):
        """

        :param SH: gravity model
        :param Nmax: max degree
        :return:
        """
        self.__SH = SH
        self.__Nmax = Nmax

        return self

    def getGeoid(self, lat, lon):
        """
        get geoid undulation of N*M points
        :param lat: geocentric latitude in degree, [dimension N]
        :param lon: geocentric longitude in degree, [dimension M]
        :return: [dimension N * M]
        """
        C, S = self.__SH.getCS(self.__Nmax)
        # C, S = np.ones(81*161), np.ones(81*161)
        C[0:3] = 0.0
        S[0:3] = 0.0

        C_normal = self.__ellipsoid.NormalGravity

        ll = len(C_normal)
        C[0:ll] -= C_normal

        geoid = self.__Har.synthesis(C, S, self.__Nmax, lat, lon, SynthesisType.Geoidheight)

        return geoid

    def __default(self):
        """
        a default gravity model "gif48" is configured to obtain the geoid undulation.
        :return:
        """
        self.__Nmax = 160
        self.__SH = Gif48().load(FileTool.get_project_dir('data/Auxiliary/GIF48.gfc'))
        # self.__SH = Gif48().load('../data/Auxiliary/ITG-Grace2010s.gfc')
        pass


def demo1():
    lat = np.arange(-89.75, 90, 0.5)
    lon = np.arange(0.25, 360, 0.5)

    geoid = GeoidUndulation(EllipsoidType.GRS80_IERS2010).getGeoid(lat, lon)

    pass


if __name__ == '__main__':
    # demo1()
    demo1()
