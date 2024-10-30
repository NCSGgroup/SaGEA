import numpy as np

from pysrc.auxiliary.aux_tool.MathTool import MathTool
from pysrc.auxiliary.preference.EnumClasses import SHCDecorrelationType, SHCDecorrelationSlidingWindowType, \
    SHCFilterType, FieldPhysicalQuantity, match_string
from pysrc.post_processing.Love_number.LoveNumber import LoveNumber
from pysrc.post_processing.convert_field_physical_quantity.ConvertSHC import ConvertSHC
from pysrc.post_processing.filter.GetSHCFilter import get_shc_decorrelation, get_shc_filter

from pysrc.post_processing.harmonic.Harmonic import Harmonic, CoreSHC, CoreGRID
from pysrc.post_processing.leakage.Addictive import Addictive
from pysrc.post_processing.leakage.BufferZone import BufferZone
from pysrc.post_processing.leakage.DataDriven import DataDriven
from pysrc.post_processing.leakage.ForwardModeling import ForwardModeling
from pysrc.post_processing.leakage.Iterative import Iterative
from pysrc.post_processing.leakage.Multiplicative import Multiplicative
from pysrc.post_processing.leakage.Scaling import Scaling
from pysrc.post_processing.leakage.ScalingGrid import ScalingGrid


def _get_filter(method: str, param: tuple = None, lmax: int = None):
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


class SHC(CoreSHC):
    def __init__(self, c, s=None):
        super().__init__(c, s)

    def __add__(self, other):
        assert issubclass(type(other), CoreSHC)

        return SHC(self.value + other.value)

    def __sub__(self, other):
        assert issubclass(type(other), CoreSHC)

        return SHC(self.value - other.value)

    def convert_type(self, from_type=None, to_type=None):
        types = list(FieldPhysicalQuantity)
        types_string = [i.name.lower() for i in types]
        types += types_string

        if from_type is None:
            from_type = FieldPhysicalQuantity.Dimensionless
        if to_type is None:
            to_type = FieldPhysicalQuantity.Dimensionless

        assert (from_type.lower() if type(
            from_type) is str else from_type) in types, f"from_type must be one of {types}"
        assert (to_type.lower() if type(
            to_type) is str else to_type) in types, f"to_type must be one of {types}"

        if from_type is None:
            from_type = FieldPhysicalQuantity.Dimensionless
        if to_type is None:
            to_type = FieldPhysicalQuantity.Dimensionless

        if type(from_type) is str:
            from_type = match_string(from_type, FieldPhysicalQuantity, ignore_case=True)
        if type(to_type) is str:
            to_type = match_string(to_type, FieldPhysicalQuantity, ignore_case=True)
        lmax = self.get_lmax()
        LN = LoveNumber()
        LN.configuration.set_lmax(lmax)

        ln = LN.get_Love_number()

        convert = ConvertSHC()
        convert.configuration.set_Love_number(ln).set_input_type(from_type).set_output_type(to_type)

        self.value = convert.apply_to(self.value)
        return self

    def to_grid(self, grid_space=None):
        if grid_space is None:
            grid_space = int(180 / self.get_lmax())

        lat, lon = MathTool.get_global_lat_lon_range(grid_space)

        lmax = self.get_lmax()
        har = Harmonic(lat, lon, lmax, option=1)

        cqlm, sqlm = self.get_cs2d()
        grid_data = har.synthesis_for_csqlm(cqlm, sqlm)
        grid = GRID(grid_data, lat, lon, option=1)

        return grid

    def filter(self, method: str, param: tuple = None):

        cqlm, sqlm = self.get_cs2d()

        filtering = _get_filter(method, param, lmax=self.get_lmax())
        cqlm_f, sqlm_f = filtering.apply_to(cqlm, sqlm)

        cs_new = []
        for i in range(np.shape(cqlm_f)[0]):
            this_cs = MathTool.cs_combine_to_triangle_1d(cqlm_f[i], sqlm_f[i])
            cs_new.append(this_cs)
        self.value = np.array(cs_new)

        return self


class GRID(CoreGRID):
    def __init__(self, grid, lat, lon, option=1):
        super().__init__(grid, lat, lon, option)

    def to_SHC(self, lmax=None):
        grid_space = self.get_grid_space()

        if lmax is None:
            lmax = int(180 / grid_space)

        lat, lon = MathTool.get_global_lat_lon_range(grid_space)

        har = Harmonic(lat, lon, lmax, option=1)

        grid_data = self.value
        cqlm, sqlm = har.analysis_for_gqij(grid_data)
        shc = SHC(cqlm, sqlm)

        return shc

    def leakage(self, method: str, basin: np.ndarray, filter_type: str, filter_params: tuple, lmax: int, times=None,
                reference: dict = None, prefilter_type: str = None, prefilter_params: tuple = None,
                shc_unfiltered: SHC = None, basin_conservation: np.ndarray = None, fm_iter_times: int = 30):
        methods_of_model_driven = (
            "addictive", "multiplicative", "scaling", "scaling_grid"
        )
        methods_of_data_driven = (
            "data_driven", "buffer_zone", "buffer", "bfz", "bf", "forward_modeling", "fm", "iterative", "iter"
        )
        methods = methods_of_data_driven + methods_of_model_driven

        method = method.lower()
        assert method in methods, f"method must be one of {methods}"

        grid_space = self.get_grid_space()
        lat, lon = MathTool.get_global_lat_lon_range(grid_space)
        har = Harmonic(lat, lon, lmax, option=1)

        filtering = _get_filter(filter_type, filter_params, lmax=lmax)

        if method in methods_of_model_driven:
            assert {"time", "model"}.issubset(set(reference.keys()))

            if method == "addictive":
                lk = Addictive()

            elif method == "multiplicative":
                lk = Multiplicative()

            elif method == "scaling":
                lk = Scaling()

            elif method == "scaling_grid":
                lk = ScalingGrid()

            else:
                assert False

            lk.configuration.set_GRACE_times(times)
            lk.configuration.set_model_times(reference["time"])
            lk.configuration.set_model(reference["model"])

        elif method in methods_of_data_driven:
            if method == "data_driven":
                assert shc_unfiltered is not None, "Data-driven requires parameter shc_unfiltered."

                lk = DataDriven()
                lk.configuration.set_cs_unfiltered(*shc_unfiltered.get_cs2d())

            elif method in ("buffer_zone", "buffer", "bfz", "bf"):
                lk = BufferZone()

            elif method in ("forward_modeling", "fm"):
                assert basin_conservation is not None, "Forward Modeling requires parameter basin_conservation."
                assert fm_iter_times is not None, "Forward Modeling requires parameter fm_iter_times."

                lk = ForwardModeling()
                lk.configuration.set_basin_conservation(basin_conservation)
                lk.configuration.set_max_iteration(fm_iter_times)

            elif method in ("iterative", "iter"):
                assert (prefilter_params is not None) and (
                        prefilter_type is not None), "Iterative requires parameter prefilter_type and prefilter_params."
                assert shc_unfiltered is not None, "Iterative requires parameter shc_unfiltered."

                lk = Iterative()

                prefilter = _get_filter(prefilter_type, prefilter_params, lmax=lmax)
                lk.configuration.set_prefilter(prefilter)
                lk.configuration.set_cs_unfiltered(*shc_unfiltered.get_cs2d())

            else:
                assert False

        else:
            assert False

        lk.configuration.set_basin(basin)
        lk.configuration.set_filter(filtering)
        lk.configuration.set_harmonic(har)

        gqij_corrected = lk.apply_to(self.value, get_grid=True)
        self.value = gqij_corrected

        return self


if __name__ == '__main__':
    pass
