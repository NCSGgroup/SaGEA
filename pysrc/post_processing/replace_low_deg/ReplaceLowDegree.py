import numpy as np

from pysrc.auxiliary.core.SHC import SHC


class ReplaceLowDegreeConfig:
    def __init__(self):
        self.replace_c00 = False

        self.replace_c10 = False
        self.replace_c11 = False
        self.replace_s11 = False

        self.replace_c20 = False
        self.replace_c30 = False

    def set_replace_c00(self, replace_or_not: bool = True):
        self.replace_c00 = replace_or_not

        return self

    def set_replace_deg1(self, replace_or_not: bool = True):
        self.replace_c10 = replace_or_not
        self.replace_c11 = replace_or_not
        self.replace_s11 = replace_or_not

        return self

    def set_replace_c20(self, replace_or_not: bool = True):
        self.replace_c20 = replace_or_not

        return self

    def set_replace_c30(self, replace_or_not: bool = True):
        self.replace_c30 = replace_or_not

        return self


class ReplaceLowDegree:
    def __init__(self):
        self.configuration = ReplaceLowDegreeConfig()

        self.low_degrees = None

    def set_low_degrees(self, low_degrees: dict):
        """
        :return:
        """
        self.low_degrees = low_degrees

        return self

    def _get_ymd_array(self, dates):
        ymd_list = []
        for i in range(len(dates)):
            year = str(dates[i].year)
            month = str(dates[i].month).rjust(2, '0')
            day = str(dates[i].day).rjust(2, '0')

            ymd_str = year + month + day
            ymd_list.append(int(ymd_str))

        return np.array(ymd_list)

    def apply_to(self, shc: SHC, begin_dates, end_dates, dev=False):
        """
        :param shc: type SHC
        :param begin_dates: list of datetime.date
        :param end_dates: list of datetime.date
        :param dev: replace deviation or not
        :return:
        """
        cqlm, sqlm = shc.get_cs2d()
        begin_times = self._get_ymd_array(begin_dates)
        end_times = self._get_ymd_array(end_dates)

        begin_times = np.array([int(begin_times[i]) for i in range(len(begin_times))])
        end_times = np.array([int(end_times[i]) for i in range(len(end_times))])

        if self.configuration.replace_c00:
            assert 'c00' in self.low_degrees.keys()

            replace_times = self._get_ymd_array(self.low_degrees['c00'][0])[:, None]
            replace_values = self.low_degrees['c00'][1]

            where = np.where(((begin_times <= replace_times) * (replace_times <= end_times)))

            cqlm[where[1], 0, 0] = replace_values[where[0]]

        if self.configuration.replace_c10:
            # assert 'c10' in self.low_degrees.keys()
            if dev:
                assert 'c10_dev' in self.low_degrees.keys()

                replace_times = self._get_ymd_array(self.low_degrees['c10_dev'][0])[:, None]
                replace_values = self.low_degrees['c10_dev'][1]

            else:
                assert 'c10' in self.low_degrees.keys()

                replace_times = self._get_ymd_array(self.low_degrees['c10'][0])[:, None]
                replace_values = self.low_degrees['c10'][1]

            where = np.where(((begin_times <= replace_times) * (replace_times <= end_times)))

            cqlm[where[1], 1, 0] = replace_values[where[0]]

        if self.configuration.replace_c11:
            # assert 'c11' in self.low_degrees.keys()
            if dev:
                assert 'c11_dev' in self.low_degrees.keys()

                replace_times = self._get_ymd_array(self.low_degrees['c11_dev'][0])[:, None]
                replace_values = self.low_degrees['c11_dev'][1]

            else:
                assert 'c11' in self.low_degrees.keys()

                replace_times = self._get_ymd_array(self.low_degrees['c11'][0])[:, None]
                replace_values = self.low_degrees['c11'][1]

            where = np.where(((begin_times <= replace_times) * (replace_times <= end_times)))

            cqlm[where[1], 1, 1] = replace_values[where[0]]

        if self.configuration.replace_s11:
            # assert 's11' in self.low_degrees.keys()
            if dev:
                assert 's11_dev' in self.low_degrees.keys()

                replace_times = self._get_ymd_array(self.low_degrees['s11_dev'][0])[:, None]
                replace_values = self.low_degrees['s11_dev'][1]

            else:
                assert 's11' in self.low_degrees.keys()

                replace_times = self._get_ymd_array(self.low_degrees['s11'][0])[:, None]
                replace_values = self.low_degrees['s11'][1]

            where = np.where(((begin_times <= replace_times) * (replace_times <= end_times)))

            sqlm[where[1], 1, 1] = replace_values[where[0]]

        if self.configuration.replace_c20:
            # assert 'c20' in self.low_degrees.keys()
            if dev:
                assert 'c20_dev' in self.low_degrees.keys()

                replace_times = self._get_ymd_array(self.low_degrees['c20_dev'][0])[:, None]
                replace_values = self.low_degrees['c20_dev'][1]

            else:
                assert 'c20' in self.low_degrees.keys()

                replace_times = self._get_ymd_array(self.low_degrees['c20'][0])[:, None]
                replace_values = self.low_degrees['c20'][1]

            where = np.where(((begin_times <= replace_times) * (replace_times <= end_times)))

            cqlm[where[1], 2, 0] = replace_values[where[0]]

        if self.configuration.replace_c30:
            # assert 'c30' in self.low_degrees.keys()
            if dev:
                assert 'c30_dev' in self.low_degrees.keys()

                replace_times = self._get_ymd_array(self.low_degrees['c30_dev'][0])[:, None]
                replace_values = self.low_degrees['c30_dev'][1]

            else:
                assert 'c30' in self.low_degrees.keys()

                replace_times = self._get_ymd_array(self.low_degrees['c30'][0])[:, None]
                replace_values = self.low_degrees['c30'][1]

            where = np.where(((begin_times <= replace_times) * (replace_times <= end_times)))

            cqlm[where[1], 3, 0] = replace_values[where[0]]

        shc_replaced = SHC(cqlm, sqlm)
        return shc_replaced


def demo():
    from pysrc.auxiliary.load_file.LoadL2SH import LoadL2SH, TimeTool
    from pysrc.auxiliary.load_file.LoadL2LowDeg import LoadLowDegree, LoadLowDegreeConfig, L2LowDegreeFileID
    import datetime

    load = LoadL2SH()
    load.configuration.set_begin_date(datetime.date(2005, 1, 1))
    load.configuration.set_end_date(datetime.date(2015, 12, 31))
    shc, dates = load.get_shc(with_dates=True)

    ave_dates = TimeTool.get_average_dates(dates[0], dates[1])
    ave_dates_with_unused_dates = TimeTool.get_average_dates(*dates)

    load_low = LoadLowDegree()
    load_low.configuration.set_file_id(L2LowDegreeFileID.TN14)

    low_degs = load_low.get_results()

    rep = ReplaceLowDegree()
    rep.configuration.set_replace_c20()
    rep.set_low_degrees(low_degs)
    rep.apply_to(shc, dates[0], dates[1])

    pass


if __name__ == '__main__':
    demo()
