from enum import Enum


class L2DataServer(Enum):
    GFZ = 1


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


class L2Release(Enum):
    RL01 = 1
    RL02 = 2
    RL03 = 3
    RL04 = 4
    RL05 = 5
    RL06 = 6
    RL061 = 61
    RL062 = 62


class L2ProductMaxDegree(Enum):
    Degree60 = 60
    Degree90 = 90
    Degree96 = 96


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


class FieldPhysicalQuantity(Enum):
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


class SHCDecorrelationType(Enum):
    PnMm = 1
    SlideWindow = 2


class SHCDecorrelationSlidingWindowType(Enum):
    Stable = 1
    Wahr2006 = 2


class LeakageMethod(Enum):
    Addictive = 1
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
