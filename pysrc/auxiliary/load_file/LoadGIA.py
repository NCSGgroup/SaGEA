import pathlib
import warnings

import numpy as np

from pysrc.auxiliary.aux_tool.TimeTool import TimeTool
from pysrc.auxiliary.core_data_class.CoreSHC import CoreSHC
from pysrc.auxiliary.load_file.LoadL2SH import load_cs
from pysrc.auxiliary.preference.EnumClasses import GIAModel
from pysrc.auxiliary.aux_tool.FileTool import FileTool


class LoadGIAConfig:
    def __init__(self):
        self.__lmax: int = 60
        self.__filepath = FileTool.get_project_dir() / 'data/GIA/GIA.ICE-6G_D.txt'
        self.__dates = None

    def set_filepath(self, filepath: pathlib.WindowsPath):
        warnings.warn('This function is deprecated, use .set_GIA_model instead', DeprecationWarning)

        self.__filepath = filepath
        return self

    def set_GIA_model(self, model: GIAModel):
        filename = None
        if model == GIAModel.Caron2018:
            filename = 'GIA.Caron_et_al_2018.txt'

        elif model == GIAModel.Caron2019:
            filename = 'GIA.Caron_Ivins_2019.txt'

        elif model == GIAModel.ICE6GC:
            filename = 'GIA.ICE-6G_C.txt'

        elif model == GIAModel.ICE6GD:
            filename = 'GIA.ICE-6G_D.txt'

        self.__filepath = FileTool.get_project_dir() / f'data/GIA/{filename}'

        return self

    def get_filepath(self):
        return self.__filepath

    def set_lmax(self, lmax: int):
        self.__lmax = lmax
        return self

    def get_lmax(self):
        return self.__lmax

    def set_dates(self, dates: list):
        self.__dates = dates
        return self

    def get_dates(self):
        return self.__dates


class LoadGIA:
    def __init__(self):
        self.configuration = LoadGIAConfig()

    def get_shc(self):
        gia_filepath = self.configuration.get_filepath()
        lmax = self.configuration.get_lmax()

        clm_trend, slm_trend = load_cs(gia_filepath, key='', lmax=lmax, lmcs_in_queue=(1, 2, 3, 4))
        shc_trend = CoreSHC(clm_trend, slm_trend)

        times_list = self.configuration.get_dates()

        year_frac = TimeTool.convert_date_format(times_list,
                                                 input_type=TimeTool.DateFormat.ClassDate,
                                                 output_type=TimeTool.DateFormat.YearFraction)

        year_frac = np.array(year_frac)

        cs_gia_trend = shc_trend.value[0]
        cs_each_times = year_frac[:, None] @ cs_gia_trend[None, :]
        shc = CoreSHC(cs_each_times)

        return shc


def demo():
    filepath = FileTool.get_project_dir() / 'data/GIA/GIA.ICE-6G_D.txt'
    lmax = 60

    load = LoadGIA()

    load.configuration.set_filepath(filepath)
    load.configuration.set_lmax(lmax)

    shc = load.get_shc()
    pass


if __name__ == '__main__':
    demo()
