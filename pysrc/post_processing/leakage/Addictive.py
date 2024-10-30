import copy
import warnings

import numpy as np

from pysrc.auxiliary.aux_tool.MathTool import MathTool
from pysrc.post_processing.leakage.BaseModelDriven import ModelDriven


class Addictive(ModelDriven):
    def apply_to(self, gqij, leakage=True, bias=True, get_grid=True):
        indexes_GRACE, indexes_model, indexes_uncorrected = self._get_indexes()

        if indexes_uncorrected:
            warnings.warn(
                f'Due to a missing model, leakage in these times will not be corrected:\n'
                f'{np.array(self.configuration.GRACE_times)[indexes_uncorrected]}'
            )

        basin_map = self.configuration.basin_map
        f_filtered = MathTool.global_integral(gqij * basin_map) / MathTool.get_acreage(basin_map)

        f_predicted = copy.deepcopy(f_filtered)

        if leakage:
            leakage_c_m = self._get_leakage()
            f_predicted[indexes_GRACE] -= leakage_c_m[indexes_model]

        if bias:
            bias_c_m = self._get_bias()
            f_predicted[indexes_GRACE] += bias_c_m[indexes_model]

        if get_grid:
            return f_predicted[:, None, None] * self.configuration.basin_map
        else:
            return f_predicted

    def format(self):
        return 'addictive'
