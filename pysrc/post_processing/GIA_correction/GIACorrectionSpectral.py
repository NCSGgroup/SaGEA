import copy
import datetime

import numpy as np

from pysrc.auxiliary.core.SHC import SHC
from pysrc.auxiliary.load_file.LoadGIA import LoadGIA
from pysrc.auxiliary.load_file.LoadL2LowDeg import LoadLowDegree
from pysrc.auxiliary.load_file.LoadL2SH import LoadL2SH
from pysrc.auxiliary.preference.EnumClasses import L2LowDegreeFileID
from pysrc.auxiliary.tools.FileTool import FileTool
from pysrc.auxiliary.tools.TimeTool import TimeTool
from pysrc.post_processing.replace_low_deg.ReplaceLowDegree import ReplaceLowDegree


class GIACorrectionSpectralConfig:
    def __init__(self):
        self.__gia_trend = None
        self.__times = None

    def set_gia_trend(self, gia: SHC):
        self.__gia_trend = gia

        return self

    def get_gia_trend(self):
        return self.__gia_trend

    def set_times(self, times):
        """
        :param times: list of datetime.date
        """
        self.__times = times
        return self

    def get_times(self):
        return self.__times


class GIACorrectionSpectral:
    def __init__(self):
        self.configuration = GIACorrectionSpectralConfig()

    def get_year_fractions(self):
        times_list = self.configuration.get_times()  # list datetime.date

        year_frac = TimeTool.convert_date_format(times_list,
                                                 input_type=TimeTool.DateFormat.ClassDate,
                                                 output_type=TimeTool.DateFormat.YearFraction)

        year_frac = np.array(year_frac)

        return year_frac

    def apply_to(self, shc):
        cs_gia_trend = self.configuration.get_gia_trend().cs
        year_frac = self.get_year_fractions()
        year_frac -= np.mean(year_frac)

        cs_each_times = year_frac[:, None] @ cs_gia_trend[None, :]

        shc_new = copy.deepcopy(shc)
        shc_new.cs -= cs_each_times

        return shc_new