import copy
import warnings

from pysrc.auxiliary.aux_tool.MathTool import MathTool
from pysrc.post_processing.leakage.BaseModelDriven import ModelDriven


class Multiplicative(ModelDriven):
    def apply_to(self, grids, get_grid=False, bias=True, leakage=True):
        """
        :param grids:
        :param get_grid:
        :param bias: if False, it will not scale the bias
        :param leakage: if False, it will not reduce the leakage_c_m
        """

        basin_map = self.configuration.basin_map

        f_filtered = MathTool.global_integral(grids.value * basin_map) / MathTool.get_acreage(basin_map)

        f_predicted = copy.deepcopy(f_filtered)
        if leakage:
            indexes_GRACE, indexes_model, indexes_uncorrected = self._get_indexes()

            if indexes_uncorrected:
                warnings.warn(
                    f'Due to a missing model, leakage in these times will not be corrected:\n'
                    f'{self.configuration.GRACE_times[indexes_uncorrected]}'
                )

            leakage_c_m = self._get_leakage() / MathTool.get_acreage(basin_map)
            f_predicted[indexes_GRACE] -= leakage_c_m[indexes_model]

        if bias:
            scale = self._get_multiplicative_scale()
            f_predicted *= scale

        if get_grid:
            return f_predicted[:, None, None] * self.configuration.basin_map
        else:
            return f_predicted

    def get_scale(self):
        return self._get_multiplicative_scale()

    def format(self):
        return 'multiplicative'
