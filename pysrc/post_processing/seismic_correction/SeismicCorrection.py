import json
import pathlib
import numpy as np
from scipy import optimize

from pysrc.auxiliary.aux_tool.TimeTool import TimeTool
from pysrc.data_class.DataClass import GRID


class SeismicCorrectionConfig:
    def __init__(self):
        self.__earthquake_list = None
        # dict, {'name': {'lat_range': [., .], 'lon_range': [., .], 'teq': [. ,], 'tau': [. ,], }}

        self.__times = None

    def set_earthquakes(self, filepath):
        """
        set earthquake events from dict or .json file
        """
        assert type(filepath) in (dict, pathlib.WindowsPath)

        if type(filepath) is pathlib.WindowsPath:
            assert filepath.name.endswith('.json')
            self.__set_earthquakes_from_json(filepath)

        elif type(filepath) is dict:
            assert False

        return self

    def __set_earthquakes_from_json(self, filepath):
        with open(filepath) as f:
            load = json.load(f)

        self.__earthquake_list = load

        return self

    def get_earthquakes(self):
        return self.__earthquake_list

    def set_times(self, times):
        """
        :param times: list of datetime.date
        """
        self.__times = times
        return self

    def get_times(self):
        return self.__times


class SeismicCorrection:
    def __init__(self):
        self.configuration = SeismicCorrectionConfig()

    def __get_year_fractions(self):
        times_list = self.configuration.get_times()  # list datetime.date

        year_frac = TimeTool.convert_date_format(times_list,
                                                 input_type=TimeTool.DateFormat.ClassDate,
                                                 output_type=TimeTool.DateFormat.YearFraction)

        year_frac = np.array(year_frac)

        return year_frac

    def apply_to(self, grid: GRID):
        earthquakes = self.configuration.get_earthquakes()
        for key in earthquakes:
            '''get lat and lon index of grid.data'''
            assert len(earthquakes[key]['lat_range']) == 2
            assert len(earthquakes[key]['lon_range']) == 2
            assert len(earthquakes[key]['teq']) in (1, 2)
            assert len(earthquakes[key]['teq']) == len(earthquakes[key]['tau'])

            lat_min, lat_max = earthquakes[key]['lat_range']
            lon_min, lon_max = earthquakes[key]['lon_range']

            lat_index = np.where((lat_min < grid.lat) & (grid.lat < lat_max))[0]
            lon_index = np.where((lon_min < grid.lon) & (grid.lon < lon_max))[0]

            teq, tau = earthquakes[key]['teq'], earthquakes[key]['tau']

            if len(earthquakes[key]['teq']) == 1:
                self.__analyse_for_once(grid, lon_index, lat_index, teq, tau)

            elif len(earthquakes[key]['teq']) == 2:
                self.__analyse_for_twice(grid, lon_index, lat_index, teq, tau)

            else:
                assert False

    def __analyse_for_once(self, grid, lon_index, lat_index, teq, tau):
        teq, tau = teq[0], tau[0]

        def fit_function1(x, c):
            return c

        def fit_function2(x, c, p):
            return c + p * (1 - np.exp(-(x - teq) / tau))

        year_fractions = self.__get_year_fractions()

        for i in lon_index:
            for j in lat_index:

                sigs = []
                for t in range(len(year_fractions)):
                    sigs.append(grid.value[t, j, i])
                sigs = np.array(sigs)

                times_before = []
                sigs_before = []
                times_after = []
                sigs_after = []
                for t in range(len(year_fractions)):
                    if year_fractions[t] < teq:
                        times_before.append(year_fractions[t])
                        sigs_before.append(sigs[t])
                    else:
                        times_after.append(year_fractions[t])
                        sigs_after.append(sigs[t])

                times_before = np.array(times_before)
                sigs_before = np.array(sigs_before)
                times_after = np.array(times_after)
                sigs_after = np.array(sigs_after)

                z1 = optimize.curve_fit(fit_function1, times_before, sigs_before)[0]
                z2 = optimize.curve_fit(fit_function2, times_after, sigs_after)[0]
                c1 = z1[0]
                c2, p = z2[0], z2[1]
                seismic = np.zeros_like(year_fractions)
                for t in range(len(year_fractions)):
                    if year_fractions[t] < teq:
                        seismic[t] = c1
                    else:
                        seismic[t] = c2 + p * (1 - np.exp(-(year_fractions[t] - teq) / tau))

                grid.value[:, j, i] -= seismic

        pass

    def __analyse_for_twice(self, grid, lon_index, lat_index, teq, tau):

        def fit_function2(x, c, p):
            return c + p * (1 - np.exp(-(x - teq1) / tau1))

        def fit_function3(x, c, p):
            return c + p * (1 - np.exp(-(x - teq2) / tau2))

        teq1, teq2 = teq[0], teq[1]
        tau1, tau2 = tau[0], tau[1]
        time_points = self.__get_year_fractions()

        for i in lon_index:
            for j in lat_index:

                sigs = []
                for t in range(len(time_points)):
                    sigs.append(grid.value[t, j, i])
                sigs = np.array(sigs)

                times_before = []
                sigs_before = []
                times_middle = []
                sigs_middle = []
                times_after = []
                sigs_after = []
                for t in range(len(time_points)):
                    if time_points[t] < teq1:
                        times_before.append(time_points[t])
                        sigs_before.append(sigs[t])
                    elif teq1 <= time_points[t] < teq2:
                        times_middle.append(time_points[t])
                        sigs_middle.append(sigs[t])
                    else:
                        times_after.append(time_points[t])
                        sigs_after.append(sigs[t])

                times_before = np.array(times_before)
                sigs_before = np.array(sigs_before)
                times_middle = np.array(times_middle)
                sigs_middle = np.array(sigs_middle)
                times_after = np.array(times_after)
                sigs_after = np.array(sigs_after)

                def fit_function1(x, c):
                    return c

                z1 = optimize.curve_fit(fit_function1, times_before, sigs_before)[0]
                z2 = optimize.curve_fit(fit_function2, times_middle, sigs_middle)[0]
                z3 = optimize.curve_fit(fit_function3, times_after, sigs_after)[0]
                c1 = z1[0]
                c2, p1 = z2[0], z2[1]
                c3, p2 = z3[0], z3[1]
                seismic = np.zeros_like(time_points)
                for t in range(len(time_points)):
                    if time_points[t] < teq1:
                        seismic[t] = c1
                    elif teq1 <= time_points[t] < teq2:
                        seismic[t] = c2 + p1 * (1 - np.exp(-(time_points[t] - teq1) / tau1))
                    else:
                        seismic[t] = c3 + p2 * (1 - np.exp(-(time_points[t] - teq2) / tau2))

                grid.value[:, j, i] -= seismic
                pass

        pass


if __name__ == '__main__':
    from pysrc.auxiliary.load_file.LoadL2SH import LoadL2SH
    from pysrc.auxiliary.preference.EnumClasses import L2InstituteType
    from pysrc.auxiliary.aux_tool.FileTool import FileTool
    import datetime

    import matplotlib.pyplot as plt

    '''load GRACE L2 SH products'''
    begin_date = datetime.date(2002, 1, 1)
    end_date = datetime.date(2015, 12, 31)
    institute = L2InstituteType.CSR
    lmax = 60

    print('loading files...')
    load = LoadL2SH()

    load.configuration.set_begin_date(begin_date)
    load.configuration.set_end_date(end_date)
    load.configuration.set_institute(institute)
    load.configuration.set_lmax(lmax)

    shc, dates = load.get_shc(with_dates=True)

    ave_dates_GRACE = TimeTool.get_average_dates(*dates)
    shc.de_background()
    '''end of loading files'''

    grid_space = 1
    grid = shc.to_grid(grid_space)

    seis = SeismicCorrection()
    seis.configuration.set_times(ave_dates_GRACE)
    seis.configuration.set_earthquakes(FileTool.get_project_dir('data/earthquake/earthquakes.json'))

    grid_data_before = grid.value.copy()
    seis.apply_to(grid)
    grid_data_after = grid.value.copy()

    plt.plot(grid_data_before[:, 125, 315])
    plt.plot(grid_data_after[:, 125, 315])
    plt.show()

    pass
