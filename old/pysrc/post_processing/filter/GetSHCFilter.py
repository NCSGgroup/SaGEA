import pysrc.auxiliary.preference.EnumClasses as Enums
from pysrc.post_processing.filter.AnisotropicGaussianHan import AnisotropicGaussianHan
from pysrc.post_processing.filter.DDK import DDK
from pysrc.post_processing.filter.Fan import Fan
from pysrc.post_processing.filter.Gaussian import Gaussian
from pysrc.post_processing.filter.PnMm import PnMm
from pysrc.post_processing.filter.SlideWindow import SlideWindow
from pysrc.post_processing.filter.VariableScale import VariableScale


def __get_shc_decorrelation(method: Enums.SHCDecorrelationType, params: tuple, ):
    """
    :param method: SHCDecorrelationType
    :param params: (n, m) for PnMm method;
                    (n, m, window length) for sliding window (stable) method;
                    (n, m, minimum window length, A, K) for sliding window (Wahr2006) method;
    """
    assert method in Enums.SHCDecorrelationType

    if method == Enums.SHCDecorrelationType.PnMm:
        dec_filter = PnMm()

        n, m = params
        dec_filter.configuration.set_n(n)
        dec_filter.configuration.set_m(m)

    elif method == Enums.SHCDecorrelationType.SlideWindowStable:
        dec_filter = SlideWindow()

        n, m, length = params
        dec_filter.configuration.set_n(n)
        dec_filter.configuration.set_m(m)
        dec_filter.configuration.set_window_length(length)

    elif method == Enums.SHCDecorrelationType.SlideWindowSwenson2006:
        dec_filter = SlideWindow()

        n, m, length, a, k = params
        dec_filter.configuration.set_n(n)
        dec_filter.configuration.set_m(m)
        dec_filter.configuration.set_window_length(length)
        dec_filter.configuration.set_param_A(a)
        dec_filter.configuration.set_param_K(k)

    else:
        assert False

    return dec_filter


def __get_shc_averaging_filter(method: Enums.SHCFilterType, params: tuple, lmax):
    """
    :param method: SHCDecorrelationType
    :param params: (radius[km], ) for Gaussian,
                    (radius_1[km], radius_2[km]) for Fan,
                    (radius_1[km], radius_2[km], m_0) for AnisotropicGaussianHan,
                    (DDKFilterType, ) for DDK
    :param lmax: lmax
    """
    assert method in Enums.SHCFilterType

    if method == Enums.SHCFilterType.Gaussian:
        shc_filter = Gaussian()

        radius = params[0]
        shc_filter.configuration.set_lmax(lmax)
        shc_filter.configuration.set_filtering_radius(radius)

    elif method == Enums.SHCFilterType.AnisotropicGaussianHan:
        shc_filter = AnisotropicGaussianHan()

        r1, r2, m0 = params
        shc_filter.configuration.set_lmax(lmax)
        shc_filter.configuration.set_filtering_params(r1, r2, m0)

    elif method == Enums.SHCFilterType.Fan:
        shc_filter = Fan()

        r1, r2 = params
        shc_filter.configuration.set_lmax(lmax)
        shc_filter.configuration.set_filtering_params(r1, r2)

    elif method == Enums.SHCFilterType.DDK:
        shc_filter = DDK()

        ddk_type = params[0]
        shc_filter.configuration.set_filter_type(ddk_type)

    else:
        return -1

    return shc_filter


def __get_grid_averaging_filter(method: Enums.SHCFilterType, params: tuple):
    assert method in Enums.GridFilterType

    if method == Enums.GridFilterType.VGC:
        '''params = (r_min, r_max, sigma2=None, vary_radius_mode: VaryRadiusWay = None, harmonic: Harmonic = None)'''

        assert 2 <= len(params) <= 5
        params = list(params)
        params += [None] * (5 - len(params))

        grid_filter = VariableScale(*params)

    else:
        return -1

    return grid_filter


def get_filter(method: Enums.SHCFilterType or Enums.SHCDecorrelationType, params: tuple = None, lmax: int = None):
    assert (method in Enums.SHCFilterType) or (method in Enums.SHCDecorrelationType) or (method in Enums.GridFilterType)

    if method in Enums.SHCFilterType:
        assert lmax is not None
        if params is None:
            if method == Enums.SHCFilterType.Gaussian:
                params = (300,)
            elif method == Enums.SHCFilterType.Fan:
                params = (300, 300,)
            elif method == Enums.SHCFilterType.AnisotropicGaussianHan:
                params = (300, 300, 25)
            elif method == Enums.SHCFilterType.DDK:
                params = (3,)

        filtering = __get_shc_averaging_filter(method=method, params=params, lmax=lmax)

    elif method in Enums.SHCDecorrelationType:
        if params is None:
            n, m, min_window, a, k = 3, 10, 10, 30, 5
            if method == Enums.SHCDecorrelationType.PnMm:
                params = (n, m)
            elif method == Enums.SHCDecorrelationType.SlideWindowStable:
                params = (n, m, min_window)
            elif method == Enums.SHCDecorrelationType.SlideWindowSwenson2006:
                params = (n, m, min_window, a, k)

        filtering = __get_shc_decorrelation(method=method, params=params)

    elif method in Enums.GridFilterType:
        if params is None:
            params = (200, 500, 0.49,)

        filtering = __get_grid_averaging_filter(method=method, params=params)

    else:
        assert False

    return filtering
