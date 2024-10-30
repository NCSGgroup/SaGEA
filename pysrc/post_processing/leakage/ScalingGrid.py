import numpy as np

from pysrc.auxiliary.aux_tool.MathTool import MathTool
from pysrc.post_processing.leakage.BaseModelDriven import ModelDriven


class ScalingGrid(ModelDriven):
    def apply_to(self, grids: np.ndarray, get_grid=False):
        grid_factor = self._get_scaling_scale_grid()

        scaled_grids = grids * grid_factor

        if get_grid:
            return scaled_grids

        else:
            basin_map = self.configuration.basin_map
            return MathTool.global_integral(scaled_grids * basin_map) / MathTool.get_acreage(basin_map)

    def format(self):
        return 'scaling grid'
