from pysrc.auxiliary.load_file.LoadL2SH import LoadL2SH
from pysrc.auxiliary.preference.EnumClasses import L2ProductType, L2InstituteType


class GMAMCorrectionConfig:
    def __init__(self):
        self.__institute_type = L2InstituteType.CSR
        self.__dates = None

    def set_GAA_institute_type(self, institute: L2InstituteType):
        self.__institute_type = institute
        return self

    def get_GAA_institute_type(self):
        return self.__institute_type

    def set_dates(self, dates):
        self.__dates = dates
        return self

    def get_dates(self):
        return self.__dates

    pass


class GMAMCorrection:
    def __init__(self):
        self.configuration = GMAMCorrectionConfig()

    def apply_to(self, shc, shc_gaa=None):
        if shc_gaa is not None:
            pass

        else:
            shc_gaa = self.__load_gaa()

        shc.value[:, 0] -= shc_gaa.cs[:0]

        return shc

    def __load_gaa(self):
        dates = self.configuration.get_dates()
        begin_date = dates[0]
        end_date = dates[-1]

        institute = self.configuration.get_GAA_institute_type()
        product = L2ProductType.GAA

        load = LoadL2SH()

        load.configuration.set_begin_date(begin_date)
        load.configuration.set_end_date(end_date)
        load.configuration.set_institute(institute)
        load.configuration.set_product_type(product)
        load.configuration.set_lmax(lmax=0)

        shc_gaa = load.get_shc()

        return shc_gaa
