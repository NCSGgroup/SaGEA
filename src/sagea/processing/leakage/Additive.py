import copy
import warnings

import numpy as np

from sagea.utils import MathTool
from sagea.processing.leakage.BaseModelDriven import ModelDriven


class Additive(ModelDriven):
    def apply_to(self, gqij, leakage=True, bias=True, get_grid=True):
        basin_map = self.configuration.basin_map
        f_filtered = MathTool.global_integral(gqij * basin_map) / MathTool.get_acreage(basin_map)

        f_predicted = copy.deepcopy(f_filtered)

        if leakage:
            leakage_c_m = self._get_leakage()
            f_predicted -= leakage_c_m

        if bias:
            bias_c_m = self._get_bias()
            f_predicted += bias_c_m

        if get_grid:
            return f_predicted[:, None, None] * self.configuration.basin_map
        else:
            return f_predicted

    def format(self):
        return 'addictive'
