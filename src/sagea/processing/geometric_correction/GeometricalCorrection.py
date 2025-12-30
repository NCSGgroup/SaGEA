import numpy as np
from tqdm import trange

from sagea.processing.geometric_correction.old.GC import GeometricalCorrection as GC
from sagea.processing.geometric_correction.old.GeoMathKit import GeoMathKit
from sagea.processing.geometric_correction.old.Setting import Assumption, FieldType

from sagea import constant


class GeometricalCorrectionConfig:
    def __init__(self):
        self.__assumption = Assumption.ActualEarth
        self.__kind = FieldType.EWH
        self.__lmax = None

        self.__lat = None
        self.__lon = None

        pass

    def set_assumption(self, assumption: Assumption):
        self.__assumption = assumption
        return self

    def get_assumption(self):
        return self.__assumption

    def set_kind(self, kind: FieldType):
        self.__kind = kind
        return self

    def get_kind(self):
        return self.__kind

    def set_lmax(self, lmax):
        self.__lmax = lmax
        return self

    def get_lmax(self):
        return self.__lmax

    def set_lat_lon(self, lat, lon):
        self.__lat, self.__lon = lat, lon
        return self

    def get_lat_lon(self):
        return self.__lat, self.__lon

    def get_lat(self):
        return self.__lat

    def get_lon(self):
        return self.__lon


class GeometricalCorrection:
    def __init__(self):
        self.configuration = GeometricalCorrectionConfig()

    def apply_to(self, cqlm, sqlm, assumption, log=False):
        assert assumption in constant.GeometricCorrectionAssumption

        if assumption == constant.GeometricCorrectionAssumption.Sphere:
            assumption = Assumption.Sphere
        elif assumption == constant.GeometricCorrectionAssumption.Ellipsoid:
            assumption = Assumption.Ellipsoid
        elif assumption == constant.GeometricCorrectionAssumption.ActualEarth:
            assumption = Assumption.ActualEarth
        else:
            assert False

        grid_space = 0.5
        lat = np.arange(-90, 90 + grid_space / 2, grid_space)
        lon = np.arange(-180 + grid_space / 2, 180 + grid_space / 2, grid_space)
        lmax = np.shape(cqlm)[1] - 1

        self.configuration.set_lat_lon(lat, lon)
        self.configuration.set_lmax(lmax)
        self.configuration.set_assumption(assumption)

        cqlm_new, sqlm_new = [], []

        if log:
            ran = trange(len(cqlm), desc="Geometrical correction")
        else:
            ran = range(len(cqlm))

        for i in ran:
            clm, slm = cqlm[i], sqlm[i]

            gc = GC().configure(
                Nmax=self.configuration.get_lmax(),
                lat=self.configuration.get_lat(),
                lon=self.configuration.get_lon(),
                assumption=self.configuration.get_assumption(),
                kind=self.configuration.get_kind()
            )

            c1d = GeoMathKit.CS_2dTo1d(clm)
            s1d = GeoMathKit.CS_2dTo1d(slm)

            cs1d_corrected = gc.setInput(GravityField=(c1d, s1d)).correct()
            clm_corrected, slm_corrected = GeoMathKit.CS_1dTo2d(cs1d_corrected[0]), GeoMathKit.CS_1dTo2d(
                cs1d_corrected[1])

            cqlm_new.append(clm_corrected)
            sqlm_new.append(slm_corrected)

        cqlm_new = np.array(cqlm_new)
        sqlm_new = np.array(sqlm_new)

        return cqlm_new, sqlm_new
