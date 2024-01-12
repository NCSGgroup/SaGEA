from pysrc.auxiliary.preference.EnumClasses import SHCDecorrelationType, SHCDecorrelationSlidingWindowType, \
    SHCFilterType
from pysrc.post_processing.filter.AnisotropicGaussianHan import AnisotropicGaussianHan
from pysrc.post_processing.filter.DDK import DDK
from pysrc.post_processing.filter.Fan import Fan
from pysrc.post_processing.filter.Gaussian import Gaussian
from pysrc.post_processing.filter.PnMm import PnMm
from pysrc.post_processing.filter.SlideWindow import SlideWindow


def get_shc_decorrelation(method: SHCDecorrelationType, params: tuple,
                          sliding_window_mode: SHCDecorrelationSlidingWindowType = None):
    """
    :param method: SHCDecorrelationType
    :param params: (n, m) for PnMm method;
                    (n, m, window length) for sliding window (stable) method;
                    (n, m, minimum window length, A, K) for sliding window (Wahr2006) method;
    :param sliding_window_mode: required if method is SHCDecorrelationType.SlideWindow
    """

    if method == SHCDecorrelationType.PnMm:
        dec_filter = PnMm()

        n, m = params
        dec_filter.configuration.set_n(n)
        dec_filter.configuration.set_m(m)

    elif method == SHCDecorrelationType.SlideWindow:
        dec_filter = SlideWindow()

        if sliding_window_mode == SHCDecorrelationSlidingWindowType.Wahr2006:
            n, m, length, a, k = params
            dec_filter.configuration.set_n(n)
            dec_filter.configuration.set_m(m)
            dec_filter.configuration.set_window_length(length)
            dec_filter.configuration.set_param_A(a)
            dec_filter.configuration.set_param_K(k)

        elif sliding_window_mode == SHCDecorrelationSlidingWindowType.Stable:
            n, m, length = params
            dec_filter.configuration.set_n(n)
            dec_filter.configuration.set_m(m)
            dec_filter.configuration.set_window_length(length)

        else:
            return -1

    else:
        return -1

    return dec_filter


def get_shc_filter(method: SHCFilterType, params: tuple, lmax):
    """
    :param method: SHCDecorrelationType
    :param params: (radius[km], ) for Gaussian,
                    (radius_1[km], radius_2[km]) for Fan,
                    (radius_1[km], radius_2[km], m_0) for AnisotropicGaussianHan,
                    (DDKFilterType, ) for DDK
    :param lmax: lmax
    """

    if method == SHCFilterType.Gaussian:
        shc_filter = Gaussian()

        radius = params[0]
        shc_filter.configuration.set_lmax(lmax)
        shc_filter.configuration.set_filtering_radius(radius)

    elif method == SHCFilterType.AnisotropicGaussianHan:
        shc_filter = AnisotropicGaussianHan()

        r1, r2, m0 = params
        shc_filter.configuration.set_lmax(lmax)
        shc_filter.configuration.set_filtering_params(r1, r2, m0)

    elif method == SHCFilterType.Fan:
        shc_filter = Fan()

        r1, r2 = params
        shc_filter.configuration.set_lmax(lmax)
        shc_filter.configuration.set_filtering_params(r1, r2)

    elif method == SHCFilterType.DDK:
        shc_filter = DDK()

        ddk_type = params[0]
        shc_filter.configuration.set_filter_type(ddk_type)

    else:
        return -1

    return shc_filter
