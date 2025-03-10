import pathlib

import numpy as np
import geopandas as gpd
import shapely.vectorized

from pysrc.data_class.GRD import GRD
from pysrc.auxiliary.aux_tool.MathTool import MathTool


class LoadShpConfig:
    def __init__(self):
        self.__filepath = None

    def set_path(self, path: pathlib.Path):
        self.__filepath = path

        return self

    def get_path(self):
        return self.__filepath


class LoadShp:
    def __init__(self, path: pathlib.Path = None):
        self.configuration = LoadShpConfig()
        if path is not None:
            self.configuration.set_path(path)
            self.__prepare()

        self.__gpf = None

    def __prepare(self):
        shp_filepath = self.configuration.get_path()
        self.__gpf = gpd.read_file(shp_filepath)
        return self

    def __load_mask(self, grid_space=None, identity_name: str = None):
        if identity_name is None:
            identity_name = "Id"

        if self.__gpf is None:
            self.__prepare()

        gdf = self.__gpf

        lat, lon = MathTool.get_global_lat_lon_range(grid_space)
        lon2d, lat2d = np.meshgrid(lon, lat)

        masks_list = []

        attr = self.get_attr(identity_name)

        for idname in np.arange(attr.size) + 1:
            bd1 = gdf[attr == idname]
            mask1 = shapely.vectorized.touches(bd1.geometry.item(), lon2d, lat2d)
            mask2 = shapely.vectorized.contains(bd1.geometry.item(), lon2d, lat2d)
            mask_of_this_id = (mask1 + mask2).astype(int)
            masks_list.append(mask_of_this_id)

        return np.array(masks_list), lat, lon

    def __load_bound(self):
        if self.__gpf is None:
            self.__prepare()

        gdf = self.__gpf
        bound = np.array([gdf.bounds.minx, gdf.bounds.maxx, gdf.bounds.miny, gdf.bounds.maxy]).T

        return bound

    def get_GRID(self, grid_space, identity_name: str = None):

        mask, lat, lon = self.__load_mask(grid_space, identity_name)

        return GRD(mask, lat, lon)

    def get_SHC(self, lmax: int, spatial_accuracy: int = None):
        if spatial_accuracy is not None:
            grid_space = spatial_accuracy
        else:
            grid_space = int(180 / lmax)

        grid = self.get_GRID(grid_space)
        shc = grid.to_SHC(lmax)
        return shc

    def get_bound(self):
        return self.__load_bound()

    def get_attr(self, identity_name: str):
        if self.__gpf is None:
            self.__prepare()

        gdf = self.__gpf
        assert identity_name in gdf.keys(), f"{identity_name} dose not exist in axes names: {gdf.keys()}"

        return gdf[identity_name]
