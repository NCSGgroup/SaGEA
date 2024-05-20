from pysrc.auxiliary.load_file.LoadL2SH import LoadL2SH
from pysrc.auxiliary.preference.EnumClasses import L2ProductType, L2InstituteType


class GAXCorrectionConfig:
    def __init__(self):
        self.__GAX_type = L2ProductType.GAD
        self.__institute_type = L2InstituteType.CSR
        self.__dates = None

    def set_GAX_type(self, gax: L2ProductType):
        self.__GAX_type = gax
        return self

    def get_GAX_type(self):
        return self.__GAX_type

    def set_institute_type(self, institute: L2InstituteType):
        self.__institute_type = institute
        return self

    def get_institute_type(self):
        return self.__institute_type

    def set_dates(self, dates):
        self.__dates = dates
        return self

    def get_dates(self):
        return self.__dates

    pass


class GAXCorrection:
    def __init__(self):
        self.configuration = GAXCorrectionConfig()

    def apply_to(self, shc, shc_gax=None):
        if shc_gax is not None:
            pass

        else:
            lmax = shc.get_lmax()

            shc_gax = self.__load_gax(lmax)

        shc.value += shc_gax.cs

        return shc

    def __load_gax(self, lmax):
        dates = self.configuration.get_dates()
        begin_date = dates[0]
        end_date = dates[-1]

        institute = self.configuration.get_institute_type()
        product = self.configuration.get_GAX_type()

        load = LoadL2SH()

        load.configuration.set_begin_date(begin_date)
        load.configuration.set_end_date(end_date)
        load.configuration.set_institute(institute)
        load.configuration.set_product_type(product)
        load.configuration.set_lmax(lmax)

        shc_gax = load.get_shc()

        return shc_gax
