import numpy as np

from pysrc.auxiliary.aux_tool.MathTool import MathTool
from pysrc.auxiliary.aux_tool.TimeTool import TimeTool
from pysrc.auxiliary.preference.EnumClasses import FieldPhysicalQuantity, match_string
from pysrc.post_processing.Love_number.LoveNumber import LoveNumber
from pysrc.post_processing.convert_field_physical_quantity.ConvertSHC import ConvertSHC
from pysrc.post_processing.filter.get_filter import get_filter

from pysrc.post_processing.harmonic.Harmonic import Harmonic, CoreSHC, CoreGRID
from pysrc.post_processing.leakage.Addictive import Addictive
from pysrc.post_processing.leakage.BufferZone import BufferZone
from pysrc.post_processing.leakage.DataDriven import DataDriven
from pysrc.post_processing.leakage.ForwardModeling import ForwardModeling
from pysrc.post_processing.leakage.Iterative import Iterative
from pysrc.post_processing.leakage.Multiplicative import Multiplicative
from pysrc.post_processing.leakage.Scaling import Scaling
from pysrc.post_processing.leakage.ScalingGrid import ScalingGrid
from pysrc.post_processing.replace_low_deg.ReplaceLowDegree import ReplaceLowDegree


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

        filtering = get_filter(method, param, lmax=self.get_lmax())
        cqlm_f, sqlm_f = filtering.apply_to(cqlm, sqlm)

        cs_new = []
        for i in range(np.shape(cqlm_f)[0]):
            this_cs = MathTool.cs_combine_to_triangle_1d(cqlm_f[i], sqlm_f[i])
            cs_new.append(this_cs)
        self.value = np.array(cs_new)

        return self

    def replace_low(self, dates_begin, dates_end, low_deg: dict,
                    deg1=True, c20=False, c30=False):
        assert len(dates_begin) == len(dates_end) == len(self.value)
        if deg1:
            c10, c11, s11 = True, True, True
        else:
            c10, c11, s11 = False, False, False

        replace_or_not = (c10, c11, s11, c20, c30)
        low_ids = ("c10", "c11", "s11", "c20", "c30")
        for i in range(len(low_ids)):
            if replace_or_not:
                assert low_ids[i] in low_deg.keys(), f"input low_deg should include key {low_ids[i]}"

        replace_low_degs = ReplaceLowDegree()
        replace_low_degs.configuration.set_replace_deg1(deg1).set_replace_c20(c20).set_replace_c30(c30)
        replace_low_degs.set_low_degrees(low_deg)

        cqlm, sqlm = replace_low_degs.apply_to(*self.get_cs2d(), begin_dates=dates_begin, end_dates=dates_end)

        self.value = MathTool.cs_combine_to_triangle_1d(cqlm, sqlm)

        return self

    def expand(self, time):
        assert not self.is_series()

        year_frac = TimeTool.convert_date_format(time,
                                                 input_type=TimeTool.DateFormat.ClassDate,
                                                 output_type=TimeTool.DateFormat.YearFraction)

        year_frac = np.array(year_frac)

        trend = self.value[0]
        value = year_frac[:, None] @ trend[None, :]
        return SHC(value)


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

        filtering = get_filter(filter_type, filter_params, lmax=lmax)

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

                prefilter = get_filter(prefilter_type, prefilter_params, lmax=lmax)
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
