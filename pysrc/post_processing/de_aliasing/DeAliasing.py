import copy

import numpy as np

from pysrc.auxiliary.aux_tool.MathTool import MathTool


def fit_function(x, a, b, c, d, e, f, g, h, ii, j, k, ll, m, n, o, p):
    """
    a: base
    b: linear trend
    c, d: annual
    e, f: semiannual
    g, h: S2 (161 days)
    i, j: P1 (171.2 days)
    k, l: S1 (322.1 days)
    m, n: K2 (1362.7 days)
    o, p: K1 (2725.4 days)
    """
    return (
            a + b * x  # linear trend
            + c * np.sin(2 * np.pi * x) + d * np.cos(2 * np.pi * x)  # annual
            + e * np.sin(2 * np.pi * x / 0.5) + f * np.cos(2 * np.pi * x / 0.5)  # semiannual
            + g * np.sin(2 * np.pi * x / (161 / 365)) + h * np.cos(2 * np.pi * x / (161 / 365))  # S2
            + ii * np.sin(2 * np.pi * x / (171.2 / 365)) + j * np.cos(2 * np.pi * x / (171.2 / 365))  # P1
            + k * np.sin(2 * np.pi * x / (322.1 / 365)) + ll * np.cos(2 * np.pi * x / (322.1 / 365))  # S1
            + m * np.sin(2 * np.pi * x / (1362.7 / 365)) + n * np.cos(2 * np.pi * x / (1362.7 / 365))  # K2
            + o * np.sin(2 * np.pi * x / (2725.4 / 365)) + p * np.cos(2 * np.pi * x / (2725.4 / 365))  # K1
    )
    pass


class DeAliasingConfig:
    def __init__(self):
        self.__de_s2 = False
        self.__de_p1 = False
        self.__de_s1 = False
        self.__de_k2 = False
        self.__de_k1 = False

    def set_de_s2(self, value: bool):
        self.__de_s2 = value

    def get_de_s2(self):
        return self.__de_s2

    def set_de_p1(self, value: bool):
        self.__de_p1 = value

    def get_de_p1(self):
        return self.__de_p1

    def set_de_s1(self, value: bool):
        self.__de_s1 = value

    def get_de_s1(self):
        return self.__de_s1

    def set_de_k2(self, value: bool):
        self.__de_k2 = value

    def get_de_k2(self):
        return self.__de_k2

    def set_de_k1(self, value: bool):
        self.__de_k1 = value

    def get_de_k1(self):
        return self.__de_k1


class DeAliasing:
    def __init__(self):
        self.configuration = DeAliasingConfig()
        self.fit_function = fit_function

    def apply_to(self, gqij, times):
        """
        :param gqij: iter
        :param times: iter, year fractions, for example, [2002., 2002.083, ...]
        """

        fitting_index_to_set_zero = []
        aliasing = (
            self.configuration.get_de_s2(), self.configuration.get_de_p1(),
            self.configuration.get_de_s1(), self.configuration.get_de_k2(),
            self.configuration.get_de_k1()
        )
        indexes = (
            [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]
        )

        for i in range(len(aliasing)):
            if aliasing[i]:
                fitting_index_to_set_zero += indexes[i]
        fitting_index_to_set_zero = np.array(fitting_index_to_set_zero)

        map_shape = np.shape(gqij[0])

        times = np.array(times)
        maps_1d = np.array([gqij[i].flatten() for i in range(len(gqij))])

        fit_result = MathTool.curve_fit(self.fit_function, times - times[0], *maps_1d)

        z = fit_result[0]
        signal_fitted_1d = np.array(
            [fit_function(times - times[0], *z[i]) for i in range(len(z))]
        )
        signal_fitted = np.array(
            [signal_fitted_1d[:, i].reshape(map_shape) for i in
             range(len(signal_fitted_1d[0]))]
        )
        residual = gqij - signal_fitted

        z_without_tide = copy.deepcopy(z)

        if len(fitting_index_to_set_zero) > 0:
            z_without_tide[:, fitting_index_to_set_zero] = 0

        signal_without_tide_1d = np.array(
            [fit_function(times - times[0], *z_without_tide[i]) for i in range(len(z_without_tide))]
        )

        signal_without_tide = np.array(
            [signal_without_tide_1d[:, i].reshape(map_shape) for i in
             range(len(signal_without_tide_1d[0]))]
        )

        return signal_without_tide + residual


if __name__ == '__main__':
    for name in (
            "s2", "p1", "s1", "k2", "k1"
    ):
        print(f"""
    def set_de_{name}(self,value: bool):
        self.__de_{name} = value

    def get_de_{name}(self):
        return self.__de_{name}
        """
              )
