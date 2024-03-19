import numpy as np

from pysrc.auxiliary.aux_tool.MathTool import MathTool
from pysrc.auxiliary.core_data_class.CoreSHC import CoreSHC
from pysrc.data_class.DataClass import SHC
from pysrc.post_processing.geometric_correction.original_files.GC import GeometricalCorrection as GC
from pysrc.post_processing.geometric_correction.original_files.GeoMathKit import GeoMathKit
from pysrc.post_processing.geometric_correction.original_files.Setting import Assumption, FieldType


class GeometricalCorrectionConfig:
    def __init__(self):
        self.__assumption = Assumption.Sphere
        self.__kind = FieldType.EWH
        self.__lmax = 60

        self.__lat = np.arange(90, -90.1, -0.5)
        self.__lon = np.arange(0, 360, 0.5)

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

    def apply_to(self, shc: CoreSHC):
        cqlm, sqlm = [], []
        for i in range(len(shc.cs)):
            print(f'geometric correction for the {i + 1}-th / {len(shc.cs)}')

            clm, slm = MathTool.cs_decompose_triangle1d_to_cs2d(shc.cs[i])

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

            cqlm.append(clm_corrected)
            sqlm.append(slm_corrected)

        return SHC(cqlm, sqlm)
