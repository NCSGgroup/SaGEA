import copy

import numpy as np
from scipy import optimize

from pysrc.data_class.DataClass import GRID
from pysrc.auxiliary.aux_tool.MathTool import MathTool


class SeismicOld:
    def __init__(self):
        self.time_points = []
        self.grids_input = []
        self.earthquakes = {}
        self.grids_correction = []
        self.grids_corrected = []
        pass

    def setSeries(self, time_points, grids):
        """
        :param time_points: year fraction
        :param grids:
        """
        self.time_points = time_points
        self.grids_input = grids
        return self

    def addEarthquakes(self, eq: dict):
        """
        :param eq: dict of dicts, supposed to include these parameters:
                    {name: {'lon': np.ndarray, 'lat': np.ndarray, 'teq': teq(list), 'tau': tau(list)}}
        """
        for name in eq.keys():
            keys_of_this_eq = eq[name].keys()
            assert {'lon', 'lat', 'teq'}.issubset(set(keys_of_this_eq))
        self.earthquakes.update(eq)
        return

    def __analyse_for_once(self, lon, lat, teq, tau):
        points = []
        time_points = np.array(self.time_points)

        gs = lon[1] - lon[0]
        for i in range(len(lon)):
            for j in range(len(lat)):
                points.append([lon[i], lat[j]])
                l, m = MathTool.getGridIndex(lon[i], lat[j], gs)

                sigs = []
                for t in range(len(time_points)):
                    sigs.append(self.grids_input[t, l, m])
                sigs = np.array(sigs)

                times_before = []
                sigs_before = []
                times_after = []
                sigs_after = []
                for t in range(len(time_points)):
                    if time_points[t] < teq:
                        times_before.append(time_points[t])
                        sigs_before.append(sigs[t])
                    else:
                        times_after.append(time_points[t])
                        sigs_after.append(sigs[t])

                times_before = np.array(times_before)
                sigs_before = np.array(sigs_before)
                times_after = np.array(times_after)
                sigs_after = np.array(sigs_after)

                def fit_function1(x, c):
                    return c

                def fit_function2(x, c, p):
                    return c + p * (1 - np.exp(-(x - teq) / tau))

                z1 = optimize.curve_fit(fit_function1, times_before, sigs_before)[0]
                z2 = optimize.curve_fit(fit_function2, times_after, sigs_after)[0]
                c1 = z1[0]
                c2, p = z2[0], z2[1]
                seismic = np.zeros_like(time_points)
                for t in range(len(time_points)):
                    if time_points[t] < teq:
                        seismic[t] = c1
                    else:
                        seismic[t] = c2 + p * (1 - np.exp(-(time_points[t] - teq) / tau))
                    # self.grids_correction[t].map[l][m] += seismic[t]
                    self.grids_correction[t, l, m] += seismic[t]
                    # self.grids_corrected[t].map[l][m] -= seismic[t]
                    self.grids_corrected[t, l, m] -= seismic[t]

        pass

    def __analyse_for_twice(self, lon, lat, teq, tau):
        teq1, teq2 = teq[0], teq[1]
        tau1, tau2 = tau[0], tau[1]
        points = []
        time_points = np.array(self.time_points)

        gs = lon[1] - lon[0]
        for i in range(len(lon)):
            for j in range(len(lat)):
                points.append([lon[i], lat[j]])
                l, m = MathTool.getGridIndex(lon[i], lat[j], gs)

                sigs = []
                for t in range(len(time_points)):
                    # sigs.append(self.grids_input[t].map[l][m])
                    sigs.append(self.grids_input[t, l, m])
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

                def fit_function2(x, c, p):
                    return c + p * (1 - np.exp(-(x - teq1) / tau1))

                def fit_function3(x, c, p):
                    return c + p * (1 - np.exp(-(x - teq2) / tau2))

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
                    self.grids_correction[t, l, m] += seismic[t]
                    self.grids_corrected[t, l, m] -= seismic[t]

        pass

    def run(self):
        self.grids_correction = copy.deepcopy(self.grids_input)
        for i in range(len(self.grids_correction)):
            # self.grids_correction[i].map = np.zeros_like(self.grids_correction[i].map)
            self.grids_correction[i] = np.zeros_like(self.grids_correction[i])
        self.grids_corrected = copy.deepcopy(self.grids_input)
        for name in self.earthquakes.keys():
            this_eq = self.earthquakes[name]

            lon = this_eq['lon']
            lat = this_eq['lat']

            if type(this_eq['teq']) in [list, tuple, np.ndarray]:
                times = len(this_eq['teq'])
            else:
                times = 1

            assert times in [1, 2]
            teq = this_eq['teq']
            tau = this_eq['tau']

            if times == 1:
                self.__analyse_for_once(lon, lat, teq, tau)

            if times == 2:
                self.__analyse_for_twice(lon, lat, teq, tau)

        pass


class SeismicConfig:
    def __init__(self):
        self.__earthquake_list = []  # [[latitudes: array, longitudes: array, time, tau], ...]

    def add_earthquake(self, earthquake: list):
        self.__earthquake_list.append(earthquake)

        return self

    def clear_earthquake(self):
        self.__earthquake_list = []

        return self


class Seismic:
    def __init__(self):
        self.configuration = SeismicConfig()

    def apply_to(self, grid: GRID):
        pass


if __name__ == '__main__':
    pass
