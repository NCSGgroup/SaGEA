#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:Yang Fan
# mailbox: yfan_cge@hust.edu.cn
# address: Huazhong University of Science and Technology, Wuhan, China
# datetime:2020/6/16 上午11:35
# software: Atmosphere dealaising product
# usage of this file:
import numpy as np

from pysrc.post_processing.geometric_correction.original_files.GeoMathKit import GeoMathKit
from pysrc.post_processing.geometric_correction.original_files.Setting import EllipsoidType


class RefEllipsoid:
    """
    Select a reference ellipsoid to do later geophysical computation like geoid undulation derivation.

    Ref:
    A http://icgem.gfz-potsdam.de/guestbook
    B https://booksite.elsevier.com/brochures/geophysics/PDFs/00054.pdf
    C https://ww2.mathworks.cn/help/map/ref/wgs84ellipsoid.html
    D IERS2010

    """

    def __init__(self, ellipsoid: EllipsoidType):
        self.type = ellipsoid
        self.__par()
        self.__calOther()
        pass

    def __par(self):
        """
        define the ellipsoid parameters according to the given model.
        :return:
        """

        if self.type == EllipsoidType.WGS84:
            '''a, semi-major axis (equatorial) [m]'''
            self.SemimajorAxis = 6378137.0
            '''1/f, inverse flatness [dimensionless]'''
            self.InverseFlattening = 298.257223563
            '''omega, angular velocity [rad s^-1]'''
            self.Omega = 7.2921151467e-5
            ''' GM, Universal grav. const. * mass of Earth [m^3 s^-2] '''
            self.GM = 3.986004418e14
            '''Stokes coefficients of normal gravity field, only C20 C40 C60 C80 and C10 is offered, see Ref A.'''
            self.NormalGravity = np.zeros(6 * 11)
            self.NormalGravity[GeoMathKit.getIndex(2, 0)] = -4.841667749599428E-04
            self.NormalGravity[GeoMathKit.getIndex(4, 0)] = 7.903037334041510E-07
            self.NormalGravity[GeoMathKit.getIndex(6, 0)] = -1.687249611016718E-09
            self.NormalGravity[GeoMathKit.getIndex(8, 0)] = 3.460524681471410E-12
            self.NormalGravity[GeoMathKit.getIndex(10, 0)] = -2.650022244590134E-15

        elif self.type == EllipsoidType.GRS80:
            '''a, semi-major axis (equatorial) [m]'''
            self.SemimajorAxis = 6378137.0
            '''1/f, inverse flatness [dimensionless]'''
            self.InverseFlattening = 298.257222101
            '''omega, angular velocity [rad s^-1]'''
            self.Omega = 7.292115e-5
            ''' GM, Universal grav. const. * mass of Earth [m^3 s^-2] '''
            self.GM = 3.986005e14
            '''Stokes coefficients of normal gravity field, only C00 C20 C40 C60 C80 and C10 is offered.'''
            self.NormalGravity = np.zeros(6 * 11)
            self.NormalGravity[GeoMathKit.getIndex(2, 0)] = -4.84166774985e-4
            self.NormalGravity[GeoMathKit.getIndex(4, 0)] = 7.90303733511e-7
            self.NormalGravity[GeoMathKit.getIndex(6, 0)] = -1.68724961151e-9
            self.NormalGravity[GeoMathKit.getIndex(8, 0)] = 3.46052468394e-10
            self.NormalGravity[GeoMathKit.getIndex(10, 0)] = -2.65002225767e-15

        elif self.type == EllipsoidType.GRS80_IERS2010:
            '''a, semi-major axis (equatorial) [m]'''
            self.SemimajorAxis = 6378136.6
            '''1/f, inverse flatness [dimensionless]'''
            self.InverseFlattening = 298.25642
            '''omega, angular velocity [rad s^-1]'''
            self.Omega = 7.292115e-5
            ''' GM, Universal grav. const. * mass of Earth [m^3 s^-2] '''
            self.GM = 3.986004418e14
            '''Stokes coefficients of normal gravity field, only C00 C20 C40 C60 C80 and C10 is offered.'''
            self.NormalGravity = np.zeros(6 * 11)
            self.NormalGravity[GeoMathKit.getIndex(2, 0)] = -4.84166774985e-4
            self.NormalGravity[GeoMathKit.getIndex(4, 0)] = 7.90303733511e-7
            self.NormalGravity[GeoMathKit.getIndex(6, 0)] = -1.68724961151e-9
            self.NormalGravity[GeoMathKit.getIndex(8, 0)] = 3.46052468394e-10
            self.NormalGravity[GeoMathKit.getIndex(10, 0)] = -2.65002225767e-15

        elif self.type == EllipsoidType.gif48:
            '''a, semi-major axis (equatorial) [m]'''
            self.SemimajorAxis = 6378136.3
            '''1/f, inverse flatness [dimensionless]'''
            self.InverseFlattening = 298.25642
            '''omega, angular velocity [rad s^-1]'''
            self.Omega = 7.292115e-5
            ''' GM, Universal grav. const. * mass of Earth [m^3 s^-2] '''
            self.GM = 3.986004415e14
            '''Stokes coefficients of normal gravity field, only C00 C20 C40 C60 C80 and C10 is offered.'''
            self.NormalGravity = np.zeros(6 * 11)
            self.NormalGravity[GeoMathKit.getIndex(2, 0)] = -4.84166774985e-4
            self.NormalGravity[GeoMathKit.getIndex(4, 0)] = 7.90303733511e-7
            self.NormalGravity[GeoMathKit.getIndex(6, 0)] = -1.68724961151e-9
            self.NormalGravity[GeoMathKit.getIndex(8, 0)] = 3.46052468394e-10
            self.NormalGravity[GeoMathKit.getIndex(10, 0)] = -2.65002225767e-15

    def __calOther(self):
        """
        to calculate all the other useful parameters
        :return:
        """

        '''gravtational constant'''
        G = 6.6742867e-11

        '''flatness'''
        self.Flattening = 1.0/self.InverseFlattening

        '''Mass of the Earth'''
        self.Mass = self.GM/G

        '''short axis'''
        self.SemiminorAxis = self.SemimajorAxis * (1 - self.Flattening)

        '''Eccentricity'''
        self.Eccentricity = np.sqrt(self.SemimajorAxis ** 2 - self.SemiminorAxis ** 2) / self.SemimajorAxis

        '''Second Eccentricity'''
        self.SecondEccentricity = np.sqrt(self.SemimajorAxis ** 2 - self.SemiminorAxis ** 2) / self.SemiminorAxis

        '''centrifugal acceleration'''
        ca = (self.Omega ** 2) * (self.SemimajorAxis ** 2) * self.SemiminorAxis / self.GM

        '''Reference: Heiskanen, W.A., and H.Moritz(1967).Physical Geodesy.Freeman, San Francisco.P.77 - 79'''
        '''# % normal gravity at the equator[ms ^ -2](2 - 105a)'''
        self.je = self.GM / (self.SemimajorAxis * self.SemiminorAxis) * (1 - 3 / 2 * ca - 3 / 14
                                                                         * (self.SecondEccentricity ** 2) * ca)
        '''# normal gravity at the pole[ms ^ -2](2 - 105b)'''
        self.jb = self.GM / (self.SemimajorAxis ** 2) * (1 + ca + 3 / 7 * (self.SecondEccentricity ** 2) * ca)

        '''# average density of the Earth: rho_ave = 5517  # kg/m^3'''
        # self.rho_ave = self.Mass / (4 * np.pi / 3 * self.SemimajorAxis ** 3)
        self.rho_ave = 5517

        self.ca = ca

        pass


def demo1():
    ell = RefEllipsoid(EllipsoidType.WGS84)
    pass


if __name__ == '__main__':
    demo1()





