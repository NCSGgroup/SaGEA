import warnings

import numpy as np

from pysrc.data_class.DataClass import GRID
from pysrc.auxiliary.aux_tool.MathTool import MathTool
from pysrc.post_processing.leakage.BaseModelDriven import ModelDriven


class Addictive(ModelDriven):
    def apply_to(self, grids: GRID, leakage=True, bias=True):
        indexes_GRACE, indexes_model, indexes_uncorrected = self._get_indexes()

        if indexes_uncorrected:
            warnings.warn(
                f'Due to a missing model, leakage in these times will not be corrected:\n'
                f'{np.array(self.configuration.GRACE_times)[indexes_uncorrected]}'
            )

        f_filtered = MathTool.global_integral(grids.data * self.configuration.basin_map)

        if leakage:
            leakage_c_m = self._get_leakage()
            f_filtered[indexes_GRACE] -= leakage_c_m[indexes_model]

        if bias:
            bias_c_m = self._get_bias()
            f_filtered[indexes_GRACE] += bias_c_m[indexes_model]

        return f_filtered

    def format(self):
        return 'addictive'


if __name__ == '__main__':
    a = np.arange(20)

    # indexes = np.arange((len(a)))
    indexes = np.arange(1, 10, 2)

    print(a[indexes])
