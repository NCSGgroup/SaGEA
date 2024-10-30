from pysrc.auxiliary.aux_tool.MathTool import MathTool
from pysrc.post_processing.leakage.BaseModelDriven import ModelDriven


class Scaling(ModelDriven):
    def apply_to(self, grids, get_grid=False):
        basin_map = self.configuration.basin_map
        f_filtered = MathTool.global_integral(grids.value * basin_map) / MathTool.get_acreage(basin_map)

        scale = self._get_scaling_scale()

        f_predicted = scale * f_filtered

        if get_grid:
            return f_predicted[:, None, None] * self.configuration.basin_map
        else:
            return f_predicted

    def get_scale(self):
        return self._get_scaling_scale()

    def format(self):
        return 'scaling'
