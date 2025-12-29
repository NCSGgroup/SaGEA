#!/use/bin/env python
# coding=utf-8
# @Author  : Shuhao Liu
# @Time    : 2025/12/29 15:34 
# @File    : constants.py

from enum import Enum


class FilterTypes(Enum):
    """
    Member name :
    Member value :
    """
    GS = "Gaussian"  # Gaussian Filter.
    DDK = "DDK"  # DDK Filter. Kusche, et al., 2007
    FAN = "Fan"  # Fan filter. Zhang et al., 2009
    HAN = "Han"  # Non-isotropic Gaussian filter by Han. Han et al., 2005.


class PhysicalDimensions(Enum):
    Geopotential = "Geopotential"
    EWH = "EWH"
    Pressure = "Pressure"
    MassDensity = "MassDensity"
    Geoid = "Geoid"
    Gravity = "Gravity"
    VerticalDisplacement = "VerticalDisplacement"


if __name__ == "__main__":
    pass
