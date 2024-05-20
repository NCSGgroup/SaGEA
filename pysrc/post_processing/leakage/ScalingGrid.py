from pysrc.data_class.DataClass import GRID
from pysrc.auxiliary.aux_tool.MathTool import MathTool
from pysrc.post_processing.leakage.BaseModelDriven import ModelDriven


class ScalingGrid(ModelDriven):
    def apply_to(self, grids: GRID, get_grids=False):
        grid_factor = self._get_scaling_scale_grid()

        scaled_grids = grids.value * grid_factor

        if get_grids:
            return scaled_grids

        else:
            return MathTool.global_integral(scaled_grids * self.configuration.basin_map)

    def format(self):
        return 'scaling grid'
