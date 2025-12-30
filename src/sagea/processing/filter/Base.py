from abc import ABC, abstractmethod

import numpy as np


class SHCFilter(ABC):
    @abstractmethod
    def apply_to(self, cqlm, sqlm):
        return cqlm, sqlm

    @staticmethod
    def _cs_to_3d_array(cqlm, sqlm):
        assert cqlm.shape == sqlm.shape
        assert len(cqlm.shape) in (2, 3)

        single = (len(cqlm.shape) == 2)

        if single:
            return np.array([cqlm]), np.array([sqlm]), single
        else:
            return cqlm, sqlm, single


def get_gaussian_weight_1d(lmax, radius_smooth, radius_earth):
    """

    :param lmax:
    :param radius_smooth: unit [m]
    :param radius_earth: unit [m]
    :return:
    """
    w = np.zeros(lmax + 1)
    b = np.log(2) / (1 - np.cos(radius_smooth / radius_earth))

    w[0] = 1
    if lmax == 0:
        return w

    w[1] = (1 + np.exp(-2 * b)) / (1 - np.exp(-2 * b)) - 1 / b
    if lmax == 1:
        return w

    else:
        for i in range(1, lmax):
            w[i + 1] = -(w[i] * (2 * i + 1)) / b + w[i - 1]

        return np.array(w)


def get_poly_func(n: int):
    """
    Dynamically generates a polynomial function.
    :param n: degree of polynomial
    :return: a polynomial function, f(x, a0, a1, a2, ...) = a0 + a1 * x + a2 * x ** 2 + ...
    """
    paras_str = ','.join(['a{}'.format(i) for i in range(n + 1)])
    return_str = '+'.join(['a{}*x**{}'.format(i, i) for i in range(n + 1)])

    func_template = 'def fit_function(x,{}):\n\treturn {}'.format(paras_str, return_str)

    exec(func_template)

    return locals()['fit_function']
