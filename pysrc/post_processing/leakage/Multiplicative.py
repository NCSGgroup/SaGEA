import warnings

from pysrc.auxiliary.aux_tool.MathTool import MathTool
from pysrc.post_processing.leakage.BaseModelDriven import ModelDriven


class Multiplicative(ModelDriven):
    def apply_to(self, grids, get_grids=False, bias=True, leakage=True):
        """
        :param grids:
        :param get_grids:
        :param bias: if False, it will not scale the bias
        :param leakage: if False, it will not reduce the leakage_c_m
        """

        f_filtered = MathTool.global_integral(grids.data * self.configuration.basin_map)

        if leakage:
            indexes_GRACE, indexes_model, indexes_uncorrected = self._get_indexes()

            if indexes_uncorrected:
                warnings.warn(
                    f'Due to a missing model, leakage in these times will not be corrected:\n'
                    f'{self.configuration.GRACE_times[indexes_uncorrected]}'
                )

            leakage_c_m = self._get_leakage()
            f_filtered[indexes_GRACE] -= leakage_c_m[indexes_model]

        if bias:
            scale = self._get_multiplicative_scale()
            f_filtered *= scale

        return f_filtered

    def get_scale(self):
        return self._get_multiplicative_scale()

    def format(self):
        return 'multiplicative'
