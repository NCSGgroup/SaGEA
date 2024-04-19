import numpy as np

from pysrc.auxiliary.aux_tool.MathTool import MathTool

curve_fit = MathTool.curve_fit


def fit_function_with_semiannual(x, a, b, c, d, e, f):
    """
    Linear Trend + Annual + Semiannual
    """
    return a + b * x + c * np.sin(2 * np.pi * x) + d * np.cos(2 * np.pi * x) + e * np.sin(4 * np.pi * x) + f * np.cos(
        4 * np.pi * x)


def fit_function_without_semiannual(x, a, b, c, d):
    """
    Linear Trend + Annual + Semiannual
    """
    return a + b * x + c * np.sin(2 * np.pi * x) + d * np.cos(2 * np.pi * x)


class OLSConfig:
    def __init__(self):
        self.__with_semiannual_or_not = True

    def set_semiannual(self, switch=True):
        self.__with_semiannual_or_not = switch
        return self

    def get_semiannual_switch(self):
        return self.__with_semiannual_or_not


class OLSFor1d:
    def __init__(self):
        self.configuration = OLSConfig()

        self.__times = None
        self.__z = None

        self.__trend = None

        self.__annual_amplitude = None
        self.__annual_phase = None  # degree

        self.__semiannual_amplitude = None
        self.__semiannual_phase = None  # degree

        self.__sigma_trend = None

        self.__sigma_annual_amplitude = None
        self.__sigma_annual_phase = None

        self.__sigma_semiannual_amplitude = None
        self.__sigma_semiannual_phase = None

        pass

    def setSignals(self, times, values):
        """
        :param times: iter, year fractions, for example, [2002., 2002.083, ...]
        :param values: iter
        """
        if self.configuration.get_semiannual_switch():
            fit_function = fit_function_with_semiannual
        else:
            fit_function = fit_function_without_semiannual

        self.__times = times
        fit_result = curve_fit(fit_function, times, values)
        # fit_result = curve_fit(fit_function, times - times[0], values)
        z = fit_result[0][0]
        self.__z = z

        sigma_z = fit_result[1]

        self.__trend = z[1]
        self.__sigma_trend = np.sqrt(sigma_z[1, 1])

        self.__annual_amplitude = np.sqrt(z[2] ** 2 + z[3] ** 2)
        self.__annual_phase = np.degrees(np.arctan(z[3] / z[2]))  # phi = arctan(C/S)

        k_annual_amp = np.array([z[2], z[3]]) / self.__annual_amplitude
        k_annual_phase = np.array([-z[2], z[3]]) / (self.__annual_amplitude ** 2)

        sigma_annual = sigma_z[2:4, 2:4]
        self.__sigma_annual_amplitude = np.sqrt(k_annual_amp @ sigma_annual @ k_annual_amp.T)
        self.__sigma_annual_phase = np.degrees(np.sqrt(k_annual_phase @ sigma_annual @ k_annual_phase.T))

        if self.configuration.get_semiannual_switch():
            self.__semiannual_amplitude = np.sqrt(z[4] ** 2 + z[5] ** 2)
            self.__semiannual_phase = np.degrees(np.arctan(z[5] / z[4]))

            k_semiannual_amp = np.array([z[4], z[5]]) / self.__semiannual_amplitude
            k_semiannual_phase = np.array([-z[5], z[4]]) / (self.__semiannual_amplitude ** 2)

            sigma_semiannual = sigma_z[4:6, 4:6]
            self.__sigma_semiannual_amplitude = np.sqrt(k_semiannual_amp @ sigma_semiannual @ k_semiannual_amp.T)
            self.__sigma_semiannual_phase = np.degrees(
                np.sqrt(k_semiannual_phase @ sigma_semiannual @ k_semiannual_phase.T))

        return self

    def get_trend(self, with_sigma=False):
        if with_sigma:
            return self.__trend, self.__sigma_trend
        else:
            return self.__trend

    def get_annual_amplitude(self, with_sigma=False):
        if with_sigma:
            return self.__annual_amplitude, self.__sigma_annual_amplitude
        else:
            return self.__annual_amplitude

    def get_annual_phase(self, with_sigma=False):
        if with_sigma:
            return self.__annual_phase, self.__sigma_annual_phase
        else:
            return self.__annual_phase

    def get_semiannual_amplitude(self, with_sigma=False):
        if with_sigma:
            return self.__semiannual_amplitude, self.__sigma_semiannual_amplitude
        else:
            return self.__semiannual_amplitude

    def get_semiannual_phase(self, with_sigma=False):
        if with_sigma:
            return self.__semiannual_phase, self.__sigma_semiannual_phase
        else:
            return self.__semiannual_phase

    def get_fitting_signal(self):
        if self.configuration.get_semiannual_switch():
            return fit_function_with_semiannual(self.__times, *self.__z)

        else:
            return fit_function_without_semiannual(self.__times, *self.__z)


class OLSForGrid:
    def __init__(self):
        self.semi_analysis = True
        self.fit_function = fit_function_with_semiannual

        self.trend = None

        self.annual_amplitude = None

        self.semiannual_amplitude = None

        pass

    def semiannual_on(self, on: bool = True):
        if not on:
            self.semi_analysis = False
            self.fit_function = fit_function_without_semiannual
        else:
            self.semi_analysis = True
            self.fit_function = fit_function_with_semiannual

        return self

    def setSignals(self, times, maps):
        """
        :param times: iter, year fractions, for example, [2002., 2002.083, ...]
        :param maps: iter
        """

        map_shape = np.shape(maps[0])

        self.trend = np.zeros_like(maps[0].flatten())
        self.annual_amplitude = np.zeros_like(maps[0].flatten())
        self.semiannual_amplitude = np.zeros_like(maps[0].flatten())

        maps_1d = [maps[i].flatten() for i in range(len(maps))]

        fit_result = curve_fit(self.fit_function, times - times[0], *maps_1d)

        z = fit_result[0]
        trend_1d = z[:, 1]
        self.trend = trend_1d.reshape(map_shape)

        annual_amp_1d = np.sqrt(z[:, 2] ** 2 + z[:, 3] ** 2)
        self.annual_amplitude = annual_amp_1d.reshape(map_shape)

        if self.semi_analysis:
            semiannual_amp_1d = np.sqrt(z[:, 4] ** 2 + z[:, 5] ** 2)
            self.semiannual_amplitude = semiannual_amp_1d.reshape(map_shape)

        return self
