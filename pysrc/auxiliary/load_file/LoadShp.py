import pathlib

import numpy as np
import geopandas as gpd
import shapely.vectorized

from pysrc.data_class.DataClass import GRID
from pysrc.auxiliary.scripts.PlotGrids import plot_grids
from pysrc.auxiliary.aux_tool.FileTool import FileTool
from pysrc.auxiliary.aux_tool.MathTool import MathTool
from pysrc.post_processing.harmonic.Harmonic import Harmonic


class ShpToMaskConfig:
    def __init__(self):
        self.__grid_space = 1
        self.__harmonic: Harmonic = None

        self.__shppath = None

    def set_grid_space(self, grid_space):
        self.__grid_space = grid_space

        return self

    def get_grid_space(self):
        return self.__grid_space

    def set_harmonic(self, harmonic: Harmonic):
        self.__harmonic = harmonic

        return self

    def get_harmonic(self):
        return self.__harmonic

    def set_shppath(self, path: pathlib.WindowsPath):
        self.__shppath = path

        return self

    def get_shppath(self):
        return self.__shppath


class ShpToMask:
    def __init__(self):
        self.configuration = ShpToMaskConfig()

    def get_basin_gridmap(self, with_whole=True):
        """

        :return: 2-dimension np.ndarray, 1 inside the basin and 0 outside.
        """
        shp_filepath = self.configuration.get_shppath()
        grid_space = self.configuration.get_grid_space()

        gdf = gpd.read_file(shp_filepath)
        lat, lon = MathTool.get_global_lat_lon_range(grid_space)

        lon2d, lat2d = np.meshgrid(lon, lat)

        mask_all = np.zeros(np.shape(lat2d))
        masks_list = [mask_all]

        for idname in np.arange(gdf.ID.size) + 1:
            bd1 = gdf[gdf.ID == idname]
            mask1 = shapely.vectorized.touches(bd1.geometry.item(), lon2d, lat2d)
            mask2 = shapely.vectorized.contains(bd1.geometry.item(), lon2d, lat2d)

            mask_of_this_id = (mask1 + mask2).astype(int)
            if with_whole:
                masks_list[0] += mask_of_this_id

            masks_list.append(mask_of_this_id)

        return np.array(masks_list)

    def get_basin_SHC(self):
        gridmap = self.get_basin_gridmap()

        har = self.configuration.get_harmonic()
        lat, lon = har.lat, har.lon
        shc = har.analysis(GRID(gridmap, lat, lon))

        return shc


def demo():
    grid_space = 0.5

    shp2mask = ShpToMask()
    shp2mask.configuration.set_shppath(FileTool.get_project_dir() / 'data/shpfiles/Danube_9_shapefiles')
    shp2mask.configuration.set_shppath(FileTool.get_project_dir() / 'data/shpfiles/Danube_9_shapefiles')
    shp2mask.configuration.set_shppath(FileTool.get_project_dir() / 'data/shpfiles/Danube_9_shapefiles')
    shp2mask.configuration.set_grid_space(grid_space)

    grid_list = shp2mask.get_basin_gridmap(with_whole=False)
    grid_whole = shp2mask.get_basin_gridmap(with_whole=True)

    lat_max_grid_index = np.min(np.where(grid_whole == 1)[0])
    lat_max = 180 - lat_max_grid_index * grid_space
    lat_max = lat_max // 5 * 5

    lat_min_grid_index = np.max(np.where(grid_whole == 1)[0])
    lat_min = 180 - lat_min_grid_index * grid_space
    lat_min = lat_min // 5 * 5

    lon_max_grid_index = np.max(np.where(grid_whole == 1)[1])
    lon_max = lon_max_grid_index * grid_space - 180
    lon_max = lon_max // 5 * 5

    lon_min_grid_index = np.min(np.where(grid_whole == 1)[1])
    lon_min = lon_min_grid_index * grid_space - 180
    lon_min = lon_min // 5 * 5

    lat, lon = MathTool.get_global_lat_lon_range(grid_space)

    for i in range(len(grid_list)):
        grid_to_plot = np.full(np.shape(grid_list[i]), np.nan)
        grid_to_plot[np.where(grid_list[i] == 1)] = 1

        plot_grids(
            grid_to_plot,
            lat=lat,
            lon=lon,
            vmin=0,
            vmax=2,
            extent=[lon_min - 5, lon_max + 5, lat_min - 5, lat_max + 5]
        )
    pass


if __name__ == '__main__':
    demo()
