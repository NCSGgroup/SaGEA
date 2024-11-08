from abc import abstractmethod

import numpy as np

from pysrc.post_processing.leakage.Base import Leakage, filter_grids
from pysrc.post_processing.filter.Base import SHCFilter
from pysrc.post_processing.harmonic.Harmonic import Harmonic
from pysrc.auxiliary.aux_tool.MathTool import MathTool


class ModelDrivenConfig:
    def __init__(self):
        self.basin_map = None
        self.basin_acreage = None
        self.filter = None
        self.harmonic = None
        self.model = None
        self.GRACE_times = None
        self.model_times = None

    def set_harmonic(self, har: Harmonic):
        self.harmonic = har
        return self

    def set_basin(self, basin: np.ndarray):
        if type(basin) is np.ndarray:
            assert basin.ndim == 2
            self.basin_map = basin

        self.basin_acreage = MathTool.get_acreage(self.basin_map)

        return self

    def set_model(self, model):
        self.model = model

        return self

    def set_filter(self, shc_filter: SHCFilter):
        self.filter = shc_filter

        return self

    def set_GRACE_times(self, times: list):
        self.GRACE_times = times

        return self

    def set_model_times(self, times: list):
        self.model_times = times

        return self


class ModelDriven(Leakage):
    """This is a base class for model-driven methods"""

    def __init__(self):
        super().__init__()
        self.configuration = ModelDrivenConfig()

    @abstractmethod
    def apply_to(self, grids, get_grid=False):
        pass

    def _get_leakage(self):
        basin = self.configuration.basin_map
        basin_outside = 1 - self.configuration.basin_map
        model_outside_basin = self.configuration.model * basin_outside

        model_outside_filtered = filter_grids(model_outside_basin, self.configuration.filter,
                                              self.configuration.harmonic)

        leakage_c_m = MathTool.global_integral(
            model_outside_filtered * self.configuration.basin_map) / MathTool.get_acreage(basin)

        return leakage_c_m

    def _get_multiplicative_scale(self):
        basin = self.configuration.basin_map

        basin_mask_filtered = \
            filter_grids(np.array([basin]), self.configuration.filter, self.configuration.harmonic)[0]

        integral_basin_mask = MathTool.global_integral(basin)
        integral_basin_mask_filtered = MathTool.global_integral(basin_mask_filtered * basin)

        return integral_basin_mask / integral_basin_mask_filtered

    def _get_bias(self):
        basin = self.configuration.basin_map

        basin_filtered = filter_grids(np.array([basin]), self.configuration.filter, self.configuration.harmonic)[0]

        bias_c_m = MathTool.global_integral(self.configuration.model * (basin - basin_filtered)) / MathTool.get_acreage(
            basin)

        return bias_c_m

    @staticmethod
    def _scale_function(x, a, b):
        return a * x + b

    def _get_scaling_scale(self):
        model_filtered = filter_grids(self.configuration.model, self.configuration.filter, self.configuration.harmonic)

        time_series_model = MathTool.global_integral(self.configuration.model * self.configuration.basin_map)
        time_series_model_filtered = MathTool.global_integral(model_filtered.value * self.configuration.basin_map)

        z = MathTool.curve_fit(self._scale_function, time_series_model_filtered, time_series_model)

        return z[0][0, 0]

    def _get_scaling_scale_grid(self):
        model_filtered = filter_grids(self.configuration.model, self.configuration.filter, self.configuration.harmonic)
        model_shape = np.shape(self.configuration.model[0])

        model_1d = np.array([self.configuration.model[i].flatten() for i in range(len(self.configuration.model))])
        model_filtered_1d = np.array([model_filtered.value[i].flatten() for i in range(len(model_filtered.value))])

        t = np.arange(len(model_1d))
        z1 = MathTool.curve_fit(self._scale_function, t, *model_1d)
        z2 = MathTool.curve_fit(self._scale_function, t, *model_filtered_1d)

        factors_grids = (z1[0][:, 0] / z2[0][:, 0]).reshape(model_shape)

        return factors_grids

    def _get_indexes(self):

        GRACE_times = self.configuration.GRACE_times
        model_times = self.configuration.model_times

        GRACE_yyyymm = [GRACE_times[i].year * 100 + GRACE_times[i].month for i in range(len(GRACE_times))]

        model_yyyymm = [model_times[i].year * 100 + model_times[i].month for i in range(len(model_times))]

        intersection = set(GRACE_yyyymm) & set(model_yyyymm)
        indexes_GRACE, indexes_model, indexes_uncorrected = [], [], []

        for i in range(len(GRACE_yyyymm)):
            if GRACE_yyyymm[i] in intersection:
                indexes_GRACE.append(i)

                indexes_model.append(model_yyyymm.index(GRACE_yyyymm[i]))

            else:
                indexes_uncorrected.append(i)

        return indexes_GRACE, indexes_model, indexes_uncorrected

    def format(self):
        return 'model driven'
