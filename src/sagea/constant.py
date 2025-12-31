#!/use/bin/env python
# coding=utf-8
# @Author  : Shuhao Liu
# @Time    : 2025/12/29 15:34 
# @File    : constant.py

from enum import Enum


class GeoConstant:
    density_earth = 5517  # unit[kg/m3]
    radius_earth = 6378136.3  # unit[m]
    GM = 3.9860044150E+14  # unit[m3/s2]

    """gas constant for dry air"""
    Rd = 287.00
    # Rd = 287.06

    '''gravity constant g defined by WMO'''
    g_wmo = 9.80665
    # g_wmo = 9.7

    ''' water density'''
    density_water = 1000.0
    # density_water = 1025.0


class SHCFilterType(Enum):
    Gaussian = "Gaussian"  # Gaussian Filter.
    DDK = "DDK"  # DDK Filter. Kusche, et al., 2007
    FAN = "Fan"  # Fan filter. Zhang et al., 2009
    HAN = "Han"  # Non-isotropic Gaussian filter by Han. Han et al., 2005.


class SHCDecorrelationType(Enum):
    PnMm = "PnMm"
    SlideWindowSwenson2006 = "SlideWindowSwenson2006"
    # SlideWindowDuan2009 = "SlideWindowDuan2009"
    SlideWindowStable = "SlideWindowStable"


class PhysicalDimension(Enum):
    Geopotential = "Geopotential"  # Dimensionless
    EWH = "EWH"  # [m]
    Pressure = "Pressure"
    MassDensity = "MassDensity"
    Geoid = "Geoid"  # [m]
    Gravity = "Gravity"
    VerticalDisplacement = "VerticalDisplacement"


class LoveNumberMethod(Enum):
    PREM = 1
    AOD04 = 2
    Wang = 3
    IERS = 4


class GeometricCorrectionAssumption(Enum):
    Sphere = 1
    Ellipsoid = 2
    ActualEarth = 3


class LeakageMethod(Enum):
    Additive = "Additive"
    Multiplicative = "Multiplicative"
    Scaling = "Scaling"
    ScalingGrid = "ScalingGrid"
    Iterative = "Iterative"
    DataDriven = "DataDriven"
    ForwardModeling = "ForwardModeling"
    BufferZone = "BufferZone"


if __name__ == "__main__":
    pass
