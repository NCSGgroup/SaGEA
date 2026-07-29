from enum import Enum


def match_string(name, obj, ignore_case=False):
    obj_list = list(obj)
    names = [obj_list[i].name for i in range(len(obj_list))]

    if ignore_case:
        names = [names[i].lower() for i in range(len(names))]
        name = name.lower()

    assert name in names

    return obj_list[names.index(name)]


class L2DataServer(Enum):
    GFZ = 1
    ITSG = 2


class L2ProductType(Enum):
    GSM = 1
    GAA = 2
    GAB = 3
    GAC = 4
    GAD = 5


class L2InstituteType(Enum):
    CSR = 1
    GFZ = 2
    JPL = 3
    COST_G = 4
    ITSG = 5


class L2Release(Enum):
    RL05 = 5
    RL06 = 6
    RL061 = 61
    RL062 = 62

    ITSGGrace2014 = 1002014
    ITSGGrace2016 = 1002016
    ITSGGrace2018 = 1002018
    ITSGGrace_operational = 2002018


class L2ProductMaxDegree(Enum):
    Degree60 = 60
    Degree90 = 90
    Degree96 = 96
    Degree120 = 120


class L2LowDegreeType(Enum):
    Deg1 = 1
    C20 = 2
    C30 = 3


class L2LowDegreeFileID(Enum):
    TN11 = 11
    TN13 = 13
    TN14 = 14


class Satellite(Enum):
    GRACE = 1
    GRACE_FO = 2


class PhysicalDimensions(Enum):
    Dimensionless = 0
    EWH = 1
    Pressure = 2
    Density = 3
    Geoid = 4
    Gravity = 5
    HorizontalDisplacementEast = 6
    HorizontalDisplacementNorth = 7
    VerticalDisplacement = 8


class LoveNumberMethod(Enum):
    PREM = 1
    AOD04 = 2
    Wang = 3
    IERS = 4


class LoveNumberType(Enum):
    VerticalDisplacement = 1
    HorizontalDisplacement = 2
    GravitationalPotential = 3
    # Do not change the order


class SHCFilterType(Enum):
    Gaussian = 1
    Fan = 2
    AnisotropicGaussianHan = 3
    DDK = 4
    VGC = 5


class GridFilterType(Enum):
    VGC = 1


class VaryRadiusWay(Enum):
    """for VGC filter"""
    sin = 1
    sin2 = 2


class SHCDecorrelationType(Enum):
    PnMm = 1
    SlideWindowSwenson2006 = 2
    SlideWindowStable = 3


class LeakageMethod(Enum):
    Additive = 1
    Multiplicative = 2
    Scaling = 3
    ScalingGrid = 4
    Iterative = 5
    DataDriven = 6
    ForwardModeling = 7
    BufferZone = 8


class GIAModel(Enum):
    Caron2018 = 1
    Caron2019 = 2
    ICE6GC = 3
    ICE6GD = 4


class BasinName(Enum):
    Amazon = 1
    Amur = 2
    Antarctica = 3
    Aral = 4
    Brahmaputra = 5
    Caspian = 6
    Colorado = 7
    Congo = 8
    Danube = 9
    Dnieper = 10
    Euphrates = 11
    Eyre = 12
    Ganges = 13
    Greenland = 14
    Indus = 15
    Lena = 16
    Mackenzie = 17
    Mekong = 18
    Mississippi = 19
    Murray = 20
    Nelson = 21
    Niger = 22
    Nile = 23
    Ob = 24
    Okavango = 25
    Orange = 26
    Orinoco = 27
    Parana = 28
    Sahara = 29
    St_Lawrence = 30
    Tocantins = 31
    Yangtze = 32
    Yellow = 33
    Yenisey = 34
    Yukon = 35
    Zambeze = 36
    Ocean = 37


class EmpiricalDecorrelationType(Enum):
    PnMm = 1
    window_stable = 2
    window_Wahr2006 = 3
    window_Duan2009 = 4


class GeometricCorrectionAssumption(Enum):
    Sphere = 1
    Ellipsoid = 2
    ActualEarth = 3
