import pathlib

import h5py
import matplotlib.pyplot as plt
import numpy as np

from pysrc.auxiliary.aux_tool.FileTool import FileTool
from pysrc.auxiliary.aux_tool.MathTool import MathTool
from pysrc.auxiliary.preference.EnumClasses import SHCDecorrelationType, SHCDecorrelationSlidingWindowType, \
    SHCFilterType, FieldPhysicalQuantity, match_string
from pysrc.post_processing.Love_number.LoveNumber import LoveNumber
from pysrc.post_processing.convert_field_physical_quantity.ConvertSHC import ConvertSHC
from pysrc.post_processing.filter.GetSHCFilter import get_shc_decorrelation, get_shc_filter

from pysrc.post_processing.harmonic.Harmonic import Harmonic, CoreSHC, CoreGRID


class SHC(CoreSHC):
    def __init__(self, c, s=None):
        super().__init__(c, s)

    def __add__(self, other):
        assert issubclass(type(other), CoreSHC)

        return SHC(self.value + other.value)

    def __sub__(self, other):
        assert issubclass(type(other), CoreSHC)

        return SHC(self.value - other.value)

    def to_grid(self, grid_space=None, from_type=None, to_type=None):

        types = list(FieldPhysicalQuantity)
        types_string = [i.name.lower() for i in types]
        types += types_string

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

        if grid_space is None:
            grid_space = int(180 / self.get_lmax())

        lat, lon = MathTool.get_global_lat_lon_range(grid_space)

        lmax = self.get_lmax()
        har = Harmonic(lat, lon, lmax, option=1)

        cqlm, sqlm = self.get_cs2d()
        grid_data = har.synthesis_for_csqlm(cqlm, sqlm)
        grid = GRID(grid_data, lat, lon, option=1)

        return grid

    def filter(self, ftype: str, param: tuple = None):
        ftypes = (
            "pnmm", "slidingwindow_wahr2006", "wahr2006", "slidingwindow_stable",
            "gaussian", "gs", "fan", "ngs", "han", "ani", "ddk",
        )

        ftype = ftype.lower()
        assert ftype in ftypes, f"ftype must be one of {ftypes}"

        cqlm, sqlm = self.get_cs2d()

        if ftype in ("pnmm",):
            if param is None:
                param = (3, 10)
            assert len(param) == 2

            filtering = get_shc_decorrelation(
                method=SHCDecorrelationType.PnMm, params=param, sliding_window_mode=None
            )

        elif ftype == ("slidingwindow_wahr2006", "wahr2006"):
            if param is None:
                param = (3, 10, 10, 30, 5)

            assert len(param) == 5

            filtering = get_shc_decorrelation(
                method=SHCDecorrelationType.SlideWindow, params=param,
                sliding_window_mode=SHCDecorrelationSlidingWindowType.Wahr2006
            )

        elif ftype == ("slidingwindow_stable",):
            if param is None:
                param = (3, 10, 5)

            assert len(param) == 3

            filtering = get_shc_decorrelation(
                method=SHCDecorrelationType.SlideWindow, params=param,
                sliding_window_mode=SHCDecorrelationSlidingWindowType.Stable
            )

        elif ftype in ("gaussian", "gs",):
            if param is None:
                param = (300,)
            assert len(param) == 1

            lmax = self.get_lmax()
            filtering = get_shc_filter(
                method=SHCFilterType.Gaussian, params=param, lmax=lmax
            )

        elif ftype in ("fan",):
            if param is None:
                param = (300, 300)
            assert len(param) == 2

            lmax = self.get_lmax()
            filtering = get_shc_filter(
                method=SHCFilterType.Fan, params=param, lmax=lmax
            )

        elif ftype in ("ngs", "han", "ani"):
            if param is None:
                param = (300, 300, 25)
            assert len(param) == 3

            lmax = self.get_lmax()
            filtering = get_shc_filter(
                method=SHCFilterType.AnisotropicGaussianHan, params=param, lmax=lmax
            )

        elif ftype in ("ddk",):
            if param is None:
                param = (3,)
            assert len(param) == 1

            lmax = self.get_lmax()
            filtering = get_shc_filter(
                method=SHCFilterType.DDK, params=param, lmax=lmax
            )

        else:
            assert False

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


if __name__ == '__main__':
    pass
