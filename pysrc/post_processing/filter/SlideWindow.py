from enum import Enum

import numpy as np

from pysrc.data_class.DataClass import SHC
from pysrc.auxiliary.aux_tool.MathTool import MathTool
from pysrc.post_processing.filter.Base import get_poly_func, SHCFilter


class SlideWindowMode(Enum):
    Stable = 1
    Wahr2006 = 2


class SlideWindowConfig:
    def __init__(self):
        self.__poly_n = 3
        self.__start_m = 10

        self.__window_mode = SlideWindowMode.Stable

        self.__window_param_A = 10  # for window mode is Wahr2006
        self.__window_param_K = 30  # for window mode is Wahr2006

        self.__window_length = 5  # if sliding mode is not stable, then it stands for the minimum length

    def set_n(self, n):
        self.__poly_n = n

        return self

    def get_n(self):
        return self.__poly_n

    def set_m(self, m):
        self.__start_m = m

        return self

    def get_m(self):
        return self.__start_m

    def set_window_mode(self, mode: SlideWindowMode):
        self.__window_mode = mode

        return self

    def get_window_mode(self):
        return self.__window_mode

    def set_param_A(self, a):
        self.__window_param_A = a

        return self

    def get_param_A(self):
        return self.__window_param_A

    def set_param_K(self, k):
        self.__window_param_K = k

        return self

    def get_param_K(self):
        return self.window_param_k

    def set_window_length(self, length):
        """
        :param length: if sliding mode is not stable, then it stands for the minimum length
        """

        self.__window_length = length

        return self

    def get_window_length(self):
        return self.__window_length


class SlideWindow(SHCFilter):
    def __init__(self):
        self.configuration = SlideWindowConfig()

    def _apply_to_cqlm(self, cqlm: np.ndarray):
        lmax = np.shape(cqlm)[1] - 1
        q = np.shape(cqlm)[0]

        window_mode = self.configuration.get_window_mode()
        if window_mode is SlideWindowMode.Wahr2006:
            A_mat = self.configuration.get_param_A()
            K = self.configuration.get_param_K()
            minimum_length = self.configuration.get_window_length()

            window_length = np.trunc(A_mat * np.exp(- np.arange(lmax + 1) / K) + 1)

            window_length[np.where(window_length < minimum_length)] = minimum_length

            window_length += window_length % 2 - 1  # to keep window length odd.

        elif window_mode is SlideWindowMode.Stable:
            window_length = np.ones((lmax + 1,)) * self.configuration.get_window_length()

        else:
            return -1

        start_m = self.configuration.get_m()
        self.fit_function = get_poly_func(self.configuration.get_n())
        for m in range(start_m, lmax + 1, 1):
            this_window_length = int(window_length[m])
            t = np.arange(this_window_length)
            A_mat = MathTool.get_design_matrix(self.fit_function, t)

            array_even = cqlm[:, m + 1::2, m]

            if np.shape(array_even)[1] < this_window_length:
                continue

            array_odd = cqlm[:, m::2, m]

            cqlm[:, m::2, m] = self._decorrelation_for_array(array_odd, this_window_length, A_mat, t, q)

            cqlm[:, m + 1::2, m] = self._decorrelation_for_array(array_even, this_window_length, A_mat, t, q)

        return cqlm

    def _decorrelation_for_array(self, array, window_length, design_mat, t, q):
        poly_n = self.configuration.get_n()

        array = array.T

        array_expand = None
        for m_pie in range(0, np.shape(array)[0] - window_length + 1):
            if array_expand is None:
                array_expand = array[m_pie:window_length + m_pie, :]
            else:
                array_expand = np.concatenate(
                    (array_expand, array[m_pie:window_length + m_pie, :]), axis=1)

        fit_params = np.linalg.pinv(design_mat) @ array_expand
        t_expand = np.array([t ** p for p in range(poly_n + 1)]).T
        fit_array = t_expand @ fit_params

        array_expand -= fit_array

        array_new_first = array_expand[:int(window_length / 2 + 1), :q]
        array_new_last = array_expand[int(window_length / 2):, -q:]
        array_new_middle = array_expand[int(window_length / 2), q:-q].reshape((-1, q))

        if np.shape(array_expand)[1] == q:
            array_new = array_expand
        else:
            array_new = np.concatenate((array_new_first, array_new_middle, array_new_last), axis=0)

        return array_new.T

    def apply_to(self, shc: SHC):
        cqlm, sqlm = shc.get_cs2d()

        length_of_cqlm = np.shape(cqlm)[0]
        csqlm = np.concatenate([cqlm, sqlm])
        csqlm = self._apply_to_cqlm(csqlm)

        cqlm_filtered = csqlm[:length_of_cqlm]
        sqlm_filtered = csqlm[length_of_cqlm:]
        return SHC(cqlm_filtered, sqlm_filtered)


def demo():
    from pysrc.post_processing.Love_number.LoveNumber import LoveNumber
    from pysrc.post_processing.convert_field_physical_quantity.ConvertSHC import ConvertSHC, FieldPhysicalQuantity
    from pysrc.auxiliary.load_file.LoadL2SH import LoadL2SH
    from pysrc.auxiliary.scripts.PlotGrids import plot_grids

    from pysrc.post_processing.harmonic.Harmonic import Harmonic
    import datetime

    load = LoadL2SH()
    load.configuration.set_begin_date(datetime.date(2005, 1, 1))
    load.configuration.set_end_date(datetime.date(2015, 12, 31))
    shc, dates = load.get_shc(with_dates=True)
    shc.de_background()

    convert = ConvertSHC()
    convert.configuration.set_output_type(FieldPhysicalQuantity.EWH)
    LN = LoveNumber()
    LN.configuration.set_lmax(60)
    ln = LN.get_Love_number()
    convert.set_Love_number(ln)
    shc = convert.apply_to(shc)

    lat, lon = MathTool.get_global_lat_lon_range(1)
    har = Harmonic(lat, lon, 60, option=1)
    grids = har.synthesis(shc)

    sw = SlideWindow()
    sw.configuration.__window_mode = SlideWindowMode.Wahr2006

    shc_filtered = sw.apply_to(shc)

    grids_filtered = har.synthesis(shc_filtered)

    plot_grids(
        np.array([grids.value[10], grids_filtered.value[10], grids_filtered.value[10]]),
        lat, lon,
        vmin=-0.2,
        vmax=0.2
    )


if __name__ == '__main__':
    demo()
