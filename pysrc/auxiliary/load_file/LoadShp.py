import pathlib

import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from shapely.prepared import prep
import shapely.vectorized

from pysrc.data_class.GRD import GRD
from pysrc.auxiliary.aux_tool.MathTool import MathTool


class LoadShp:
    def __init__(self, path: pathlib.Path):
        self.__gpf = gpd.read_file(path)

    def __load_mask_old(self, grid_space=None, identity_name: str = None):
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

    def __load_mask(self, grid_space):
        """
        Generate individual global grid masks for each independent feature in the Shapefile

        params：
        shp_path: pathlib.Path
        grid_space: int ot float, space of global grid

        return:
        list: list[np.ndarray], list of individual global grid masks
        """

        gdf = self.__gpf

        gdf.to_crs(epsg=4326, inplace=True)
        geometries = gdf.geometry.tolist()

        output_width, output_height = int(round(360 / grid_space, 0)), int(round(180 / grid_space, 0)),

        mask_global_list = []
        lat, lon = MathTool.get_global_lat_lon_range(grid_space)

        for idx, geom in enumerate(geometries):
            mask_global = np.zeros((output_height, output_width), dtype=np.uint8)

            g_minx, g_miny, g_maxx, g_maxy = geom.bounds
            lat_index = np.where((lat < g_maxy) * (lat > g_miny))[0]
            lon_index = np.where((lon < g_maxx) * (lon > g_minx))[0]

            prepared_geom = prep(geom)

            for i in lat_index:
                this_lat = lat[i]
                for j in lon_index:
                    this_lon = lon[j]
                    if prepared_geom.contains(Point(this_lon, this_lat)):
                        mask_global[i, j] = 1

            mask_global_list.append(mask_global)

        return mask_global_list, lat, lon

    def __load_bound(self):
        if self.__gpf is None:
            self.__prepare()

        gdf = self.__gpf
        bound = np.array([gdf.bounds.minx, gdf.bounds.maxx, gdf.bounds.miny, gdf.bounds.maxy]).T

        return bound

    def get_GRID(self, grid_space, identity_name: str = None):

        mask, lat, lon = self.__load_mask(grid_space)

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
