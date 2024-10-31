from pysrc.auxiliary.preference.EnumClasses import SHCFilterType, SHCDecorrelationType, \
    SHCDecorrelationSlidingWindowType
from pysrc.post_processing.filter.GetSHCFilter import get_shc_filter, get_shc_decorrelation


def get_filter(method: str, param: tuple = None, lmax: int = None):
    methods = (
        "pnmm", "slidingwindow_wahr2006", "wahr2006", "slidingwindow_stable",
        "gaussian", "gs", "fan", "ngs", "han", "ani", "ddk",
    )

    method = method.lower()
    assert method in methods, f"method must be one of {methods}"
    if method in ("pnmm",):
        if param is None:
            param = (3, 10)
        assert len(param) == 2

        filtering = get_shc_decorrelation(
            method=SHCDecorrelationType.PnMm, params=param, sliding_window_mode=None
        )

    elif method == ("slidingwindow_wahr2006", "wahr2006"):
        if param is None:
            param = (3, 10, 10, 30, 5)

        assert len(param) == 5

        filtering = get_shc_decorrelation(
            method=SHCDecorrelationType.SlideWindow, params=param,
            sliding_window_mode=SHCDecorrelationSlidingWindowType.Wahr2006
        )

    elif method == ("slidingwindow_stable",):
        if param is None:
            param = (3, 10, 5)

        assert len(param) == 3

        filtering = get_shc_decorrelation(
            method=SHCDecorrelationType.SlideWindow, params=param,
            sliding_window_mode=SHCDecorrelationSlidingWindowType.Stable
        )

    elif method in ("gaussian", "gs",):
        if param is None:
            param = (300,)
        assert len(param) == 1

        filtering = get_shc_filter(
            method=SHCFilterType.Gaussian, params=param, lmax=lmax
        )

    elif method in ("fan",):
        if param is None:
            param = (300, 300)
        assert len(param) == 2

        filtering = get_shc_filter(
            method=SHCFilterType.Fan, params=param, lmax=lmax
        )

    elif method in ("ngs", "han", "ani"):
        if param is None:
            param = (300, 300, 25)
        assert len(param) == 3

        filtering = get_shc_filter(
            method=SHCFilterType.AnisotropicGaussianHan, params=param, lmax=lmax
        )

    elif method in ("ddk",):
        if param is None:
            param = (3,)
        assert len(param) == 1

        filtering = get_shc_filter(
            method=SHCFilterType.DDK, params=param, lmax=lmax
        )

    else:
        assert False

    return filtering
