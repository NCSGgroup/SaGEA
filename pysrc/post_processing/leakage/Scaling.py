from pysrc.auxiliary.aux_tool.MathTool import MathTool
from pysrc.post_processing.leakage.BaseModelDriven import ModelDriven


class Scaling(ModelDriven):
    def apply_to(self, grids):
        f_filtered = MathTool.global_integral(grids.data * self.configuration.basin_map)

        scale = self._get_scaling_scale()

        return scale * f_filtered

    def get_scale(self):
        return self._get_scaling_scale()

    def format(self):
        return 'scaling'
