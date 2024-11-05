"""
@Company: CGE-HUST, Wuhan, China
@Author: Yang Fan
@Contact: yfan_cge@hust.edu.cn
@Modify Time:2022/6/1
@Description:
"""

from pysrc.auxiliary.aux_tool.FileTool import FileTool
from pysrc.post_processing.geometric_correction.old.Setting import FieldType, Assumption, Constants, HarAnalysisType, \
    LoveNumberType, EllipsoidType, DataType, \
    SynthesisType
from pysrc.post_processing.geometric_correction.old.GeoMathKit import GeoMathKit
from pysrc.post_processing.geometric_correction.old.GeoidUndulation import GeoidUndulation
from pysrc.post_processing.geometric_correction.old.Harmonic import Harmonic
from pysrc.post_processing.geometric_correction.old.LoveNumber import LoveNumber
from pysrc.post_processing.geometric_correction.old.RefEllipsoid import RefEllipsoid
from netCDF4 import Dataset

import numpy as np


class ReadNC:
    """
    Only work for reading ECMWF temperature/humidity/surface pressure/geo-potential

    """

    def __init__(self):
        self.__nameOfData = {DataType.TEMP: "t",
                             DataType.SHUM: "q",
                             DataType.PSFC: "sp",
                             DataType.PHISFC: "z"}

        self.__DataType = None
        self.__nc = None
        pass

    def setPar(self, file, datatype: DataType):
        self.__DataType = datatype
        self.__nc = Dataset(file)
        return self

    def read(self, seqN: int = 0):
        """
        read data from NC
        :param seqN: the sequence number of given date-time, only for the surface data
        :return:
        """
        assert isinstance(seqN, int)
        if self.__DataType != DataType.PSFC:
            assert seqN == 0

        try:
            '''multi-level data'''
            level = self.__nc.variables['level'].size
            res = self.__nc.variables[self.__nameOfData[self.__DataType]][seqN, :, :, :]
        except Exception as e:
            '''surface data'''
            res = self.__nc.variables[self.__nameOfData[self.__DataType]][seqN, :, :]
        finally:
            longitude = self.__nc.variables['longitude'][:]
            latitude = self.__nc.variables['latitude'][:]

        return np.array(res), np.array(latitude), np.array(longitude)


class ForwardModel:
    """
    this class is used to convert EWH or Pressure field into geo-potential Spherical Harmonic under a certain assumption
    """

    def __init__(self, orography, undulation, elliposid: RefEllipsoid, loveNumber: LoveNumber):

        self.__ellipsoid = elliposid
        self.__orography = orography
        self.__undulation = undulation
        self.__loveNumber = loveNumber
        self.__assumption = None
        pass

    def setAssumption(self, assumption=Assumption.Sphere):
        self.__assumption = assumption
        return self

    def getCS(self, lat, lon, field, maxDeg: int, kind=FieldType.EWH):
        """
        Set pressure or other physical fields input and carry out surface ellipsoidal integral
        :param pressure: pressure field or EWH field or ...  [N*M]
        :param maxDeg: up to given max degree/order that output Stokes coefficients will get
        :return:
        """
        assert kind in FieldType
        lonMesh, latMesh = np.meshgrid(lon, lat)
        lonMesh = lonMesh.flatten()
        latMesh = latMesh.flatten()
        '''reference ellipsoid + geoid undulation + orography'''
        z = self._getHeight(latMesh, self.__orography / Constants.g_wmo)

        if self.__assumption == Assumption.ActualEarth:

            r = self._getR(latMesh) + self.__undulation + z

        elif self.__assumption == Assumption.Ellipsoid:

            r = self._getR(latMesh)

        else:

            r = 0
        # r = self._getR(latMesh) # consider the topography or not
        ar = r / self.__ellipsoid.SemimajorAxis

        gr = self._getG(latMesh, z)
        # gr = Constants.g_wmo

        deltaI = []
        iniPower = ar

        if self.__assumption == Assumption.Sphere:
            if kind == FieldType.Pressure:
                deltaI = [field / gr]
            else:
                deltaI = [field]
        else:

            for i in range(maxDeg + 1):
                I_lev = np.zeros(np.size(ar))
                iniPower = iniPower * ar
                # I_lev = np.power(ar, i + 2) * pressure / self._getG(latMesh, z)
                if kind == FieldType.Pressure:
                    I_lev = iniPower * field / gr
                else:
                    I_lev = iniPower * field
                deltaI.append(I_lev)

        Pnm = GeoMathKit.getPnm(lat, maxDeg, 1)  # run for once is enough

        hm = Harmonic(self.__loveNumber, Parallel=-1).setLoveNumMethod(LoveNumberType.Wang)

        if kind == FieldType.Pressure:
            cnm, snm = hm.analysis(Nmax=maxDeg, Inner=deltaI, lat=lat, lon=lon, Pnm=Pnm,
                                   kind=HarAnalysisType.InnerIntegral)
        else:
            cnm, snm = hm.analysis(Nmax=maxDeg, Inner=deltaI, lat=lat, lon=lon, Pnm=Pnm,
                                   kind=HarAnalysisType.InnerIntegral_EWH)

        return cnm, snm

    @DeprecationWarning
    def getCS2(self, lat, lon, field, maxDeg: int, kind=FieldType.EWH):
        """
        Set pressure or other physical fields input and carry out surface ellipsoidal integral
        :param pressure: pressure field or EWH field or ...  [N*M]
        :param maxDeg: up to given max degree/order that output Stokes coefficients will get
        :return:
        """

        lonMesh, latMesh = np.meshgrid(lon, lat)
        lonMesh = lonMesh.flatten()
        latMesh = latMesh.flatten()
        '''reference ellipsoid + geoid undulation + orography'''
        z = self._getHeight(latMesh, self.__orography / Constants.g_wmo)

        if self.__assumption == Assumption.ActualEarth:

            r = self._getR(latMesh) + self.__undulation + z

        elif self.__assumption == Assumption.Ellipsoid:

            r = self._getR(latMesh)

        else:

            r = 0
        # r = self._getR(latMesh) # consider the topography or not
        ar = r / self.__ellipsoid.SemimajorAxis

        # gr = self._getG(latMesh, z)
        gr = Constants.g_wmo

        deltaI = []
        iniPower = ar

        if self.__assumption == Assumption.Sphere:
            if kind == FieldType.Pressure:
                deltaI = [field / gr]
            else:
                deltaI = [field]
        else:

            for i in range(maxDeg + 1):
                I_lev = np.zeros(np.size(ar))
                iniPower = iniPower * ar
                # I_lev = np.power(ar, i + 2) * pressure / self._getG(latMesh, z)
                if kind == FieldType.Pressure:
                    I_lev = iniPower * field / gr
                else:
                    I_lev = iniPower * field
                deltaI.append(I_lev)

        Pnm = GeoMathKit.getPnm(lat, maxDeg, 1)  # run for once is enough

        hm = Harmonic(self.__loveNumber, Parallel=-1).setLoveNumMethod(LoveNumberType.Wang)

        if kind == FieldType.Pressure:
            cnm, snm = hm.analysis(Nmax=maxDeg, Inner=deltaI, lat=lat, lon=lon, Pnm=Pnm,
                                   kind=HarAnalysisType.InnerIntegral)
        else:
            cnm, snm = hm.analysis(Nmax=maxDeg, Inner=deltaI, lat=lat, lon=lon, Pnm=Pnm,
                                   kind=HarAnalysisType.InnerIntegral_EWH)

        return cnm, snm

    def _getR(self, lat):
        # % Input:
        # % lat...Latitudes in degree
        # % ellipsoid...ellipsoidal
        # Parameters
        # % Output:
        # % r...Latitutde - dependent Radius

        B = lat * np.pi / 180  # latitutde in [rad]
        R = self.__ellipsoid.SemimajorAxis
        e = self.__ellipsoid.Eccentricity

        r = R * np.sqrt(1 - (e ** 2) * (np.sin(B) ** 2))

        return r

    def _getHeight(self, theta, H):
        """

        :param theta: latitude in degree
        :param H: geopoetntial height
        :return:
        """
        B = (90 - theta) * np.pi / 180  # co-latitude in [rad]

        '''AOD RL06 document: get the real height from geopotential height'''
        z = (1 - 0.002644 * np.cos(2 * B)) * H + (1 - 0.0089 * np.cos(2 * B)) * np.power(H, 2) / 6.245e6

        return z

    def _getG(self, theta, z):
        """

        :param theta:
        :param z:
        :return:
        """
        '''Boy and Chao (2005), Precise evaluation of atmospheric loading effects
        on Earth’s time-variable gravity field. Eq. 18 and 19 '''
        # notice: the input z must be the real height

        B = (90 - theta) * np.pi / 180  # co-latitutde in [rad]

        R = self.__ellipsoid.SemimajorAxis
        g_theta_z = self.__ellipsoid.je * (1 + 5.2885e-3 * np.power(np.cos(B), 2) - 5.9e-6 * np.power(np.cos(2 * B), 2)) \
                    * (1 - 2 * (1.006803 - 0.060706 * np.power(np.cos(B), 2)) * z / R + 3 * (z / R) ** 2)

        return g_theta_z


class GeometricalCorrection:
    """
    make a correction
    """

    def __init__(self):
        self.__assumption = None
        self.__gf = None
        self.__kind = None
        self.__Nmax = None
        self.__lat, self.__lon = None, None
        pass

    def configure(self, Nmax, lat, lon, assumption=Assumption.Sphere, kind=FieldType.EWH):
        self.__Nmax = Nmax
        self.__lat, self.__lon = lat, lon
        # self.__gf = GravityField
        self.__assumption = assumption
        self.__kind = kind
        self.__prepare()
        return self

    def setInput(self, GravityField):
        self.__gf = GravityField
        return self

    def __prepare(self):
        '''preparation'''
        elltype = EllipsoidType.GRS80_IERS2010
        ell = RefEllipsoid(elltype)
        undulation = GeoidUndulation(elltype).getGeoid(self.__lat, self.__lon).flatten()

        LN = LoveNumber(FileTool.get_project_dir('data/auxiliary/'))
        PHISFC = FileTool.get_project_dir('data/topography/PHISFC_ERA5_invariant.nc', relative=True)
        orography = ReadNC().setPar(PHISFC, DataType.PHISFC).read()[0].flatten()
        self.__Pnm = GeoMathKit.getPnm(self.__lat, self.__Nmax, 1)

        self.__fm = ForwardModel(orography=orography, undulation=undulation, elliposid=ell,
                                 loveNumber=LN).setAssumption(assumption=self.__assumption)

        self.__HM = Harmonic(LN).setLoveNumMethod(LoveNumberType.Wang)

        pass

    def correct(self, iterMax=4, Vmax=2.5, Vmin=0):
        lat, lon = self.__lat, self.__lon
        Pnm, fm, HM = self.__Pnm, self.__fm, self.__HM

        CnmT0, SnmT0 = self.__gf[0], self.__gf[1]
        CnmT, SnmT = CnmT0, SnmT0

        # grid = []
        # CS = []

        SynType = SynthesisType[self.__kind.name]

        '''surface mass derived from true CS for spherical Earth'''
        gg = HM.synthesis(Cnm=CnmT, Snm=SnmT, lat=lat, lon=lon, Nmax=self.__Nmax, kind=SynType)

        # grid.append(gg)
        # CS.append([CnmT, SnmT])

        for iter in range(iterMax):
            # print('Iteration No: %s' % iter)
            '''reference'''
            CnmR, SnmR = fm.getCS(lat=lat, lon=lon, field=gg.flatten(), maxDeg=self.__Nmax, kind=self.__kind)

            xr_C, xr_S = CnmT / CnmR, SnmT / SnmR

            '''special treatment of the scale factor'''
            xr_C[np.isnan(xr_C)] = 1
            xr_S[np.isnan(xr_S)] = 1
            # if iter > 0:
            #     Vmax = 0.7
            #     Vmin = 0.2
            # xr_C[xr_C > Vmax] =1
            # xr_C[xr_C < Vmin] = 1
            # xr_S[xr_S > Vmax] =1
            # xr_S[xr_S < Vmin] = 1
            xr_C[np.fabs(xr_C) > Vmax] = 1
            xr_C[np.fabs(xr_C) < Vmin] = 1
            xr_S[np.fabs(xr_S) > Vmax] = 1
            xr_S[np.fabs(xr_S) < Vmin] = 1

            '''update CnmT'''
            CnmT, SnmT = xr_C * CnmT0, xr_S * SnmT0
            SnmT[np.isnan(SnmT)] = 0
            CnmT[np.isnan(CnmT)] = 0
            gg = HM.synthesis(Cnm=CnmT, Snm=SnmT, lat=lat, lon=lon, Nmax=self.__Nmax, kind=SynType)
            # grid.append(gg)
            # CS.append([CnmT, SnmT])

        return CnmT, SnmT
