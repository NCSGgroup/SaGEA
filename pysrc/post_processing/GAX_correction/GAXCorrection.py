from pysrc.auxiliary.preference.EnumClasses import L2ProductType


class GAXCorrectionConfig:
    def __init__(self):
        self.__GAX_type = L2ProductType.GAD
        self.__dates = None

    def set_GAX_type(self, gax: L2ProductType):
        self.__GAX_type = gax
        return self

    def get_GAX_type(self):
        return self.__GAX_type

    def set_dates(self, dates):
        self.__dates = dates

    pass


class GAXCorrection():
    pass
