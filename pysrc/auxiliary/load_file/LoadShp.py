import pathlib

import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
from shapely.ops import transform
from shapely.prepared import prep
import shapely.vectorized
from pyproj import Transformer, CRS

from pysrc.data_class.GRD import GRD
from pysrc.auxiliary.aux_tool.MathTool import MathTool


class LoadShp:
    def __init__(self, path: pathlib.Path):
        self.__gpf = gpd.read_file(path)

    def __load_mask_old(self, grid_space=None, identity_name: str = None):
        if identity_name is None:
            identity_name = "Id"

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

    def __load_mask(self, grid_space, mode="polygon", around_polar=False):
        """
        Generate individual global grid masks for each independent feature in the Shapefile

        params：
        shp_path: pathlib.Path
        grid_space: int ot float, space of global grid
        mode: str, "point" or "polygon"
        around_polar: "North", "South", or None, EXPERIMENTAL.

        return:
        list: list[np.ndarray], list of individual global grid masks
            if mode == "point": all points are inside one mask, and length of returned list is 1.
            elif mode == "polygon": each polygon is inside each mask,
                                    and length of returned list equals to the number of polygons.
            elif mode == "multipoint":
        """
        assert mode in ["point", "polygon", "multipoint"]

        gdf = self.__gpf

        # if gdf.crs is not None and gdf.crs.srs != "EPSG:4326":
        #     gdf.to_crs(epsg=4326, inplace=True)

        # if gdf.crs is None:
        #     gdf.to_crs(epsg=4326, inplace=True)

        # if around_polar == "South":
        #     gdf = gdf.to_crs(epsg=3031)

        geometries = gdf.geometry.tolist()

        output_width, output_height = int(round(360 / grid_space, 0)), int(round(180 / grid_space, 0)),

        lat, lon = MathTool.get_global_lat_lon_range(grid_space)

        if mode == "polygon":
            mask_list = []
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

                mask_list.append(mask_global)

        elif mode == "point":
            mask_global = np.zeros((output_height, output_width), dtype=np.uint8)

            cplygn = Polygon([(geometries[i].x, geometries[i].y) for i in range(len(geometries))])

            g_minx, g_miny, g_maxx, g_maxy = cplygn.bounds
            lat_index = np.where((lat < g_maxy) * (lat > g_miny))[0]
            lon_index = np.where((lon < g_maxx) * (lon > g_minx))[0]

            prepared_geom = prep(cplygn)

            for i in lat_index:
                this_lat = lat[i]
                for j in lon_index:
                    this_lon = lon[j]
                    if prepared_geom.contains(Point(this_lon, this_lat)):
                        mask_global[i, j] = 1

            mask_list = [mask_global]

        elif mode == "multipoint":
            mask_global = np.zeros((output_height, output_width), dtype=np.uint8)

            point_list = list(geometries[0].geoms)

            if around_polar == "South":
                """experimental"""

                crs_antarctic = CRS("EPSG:3031")
                transformer = Transformer.from_crs("EPSG:4326", crs_antarctic, always_xy=True)

                # projected_coords = [transformer.transform(p.x, p.y) for p in point_list]
                projected_coords = [transformer.transform(p.x + 180, p.y) for p in point_list]

                polygon_proj = Polygon(projected_coords)

                transformer_back = Transformer.from_crs(crs_antarctic, "EPSG:4326", always_xy=True)
                cplygn = transform(transformer_back.transform, polygon_proj)

            else:
                cplygn = Polygon([(point_list[i].x, point_list[i].y) for i in range(len(point_list))])

            # cplygn = Polygon([(point_list[i].x, point_list[i].y) for i in range(len(point_list))])

            g_minx, g_miny, g_maxx, g_maxy = cplygn.bounds
            lat_index = np.where((lat < g_maxy) * (lat > g_miny))[0]
            lon_index = np.where((lon < g_maxx) * (lon > g_minx))[0]

            prepared_geom = prep(cplygn)

            for i in lat_index:
                this_lat = lat[i]
                for j in lon_index:
                    this_lon = lon[j]
                    if prepared_geom.contains(Point(this_lon, this_lat)):
                        mask_global[i, j] = 1

            if around_polar == "South":
                """experimental"""
                mask_global_new = np.zeros_like(mask_global)
                mask_global_new[:, int(180 / grid_space):] = mask_global[:, :int(180 / grid_space)]
                mask_global_new[:, :int(180 / grid_space)] = mask_global[:, int(180 / grid_space):]

                mask_global = mask_global_new

            mask_list = [mask_global]


        else:
            assert False

        return mask_list, lat, lon

    def __multipoint_to_one_mask(self, grid_space):
        pass

    def __load_bound(self):
        gdf = self.__gpf
        bound = np.array([gdf.bounds.minx, gdf.bounds.maxx, gdf.bounds.miny, gdf.bounds.maxy]).T

        return bound

    def get_GRID(self, grid_space, mode="polygon", around_polar=None):
        assert mode in ["point", "polygon", "multipoint"]

        mask, lat, lon = self.__load_mask(grid_space, mode=mode, around_polar=around_polar)

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
        gdf = self.__gpf
        assert identity_name in gdf.keys(), f"{identity_name} dose not exist in axes names: {gdf.keys()}"

        return gdf[identity_name]

    def keys(self):
        gdf = self.__gpf
        return gdf.keys()
