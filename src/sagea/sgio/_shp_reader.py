#!/use/bin/env python
# coding=utf-8
# @Author  : Shuhao Liu
# @Time    : 2025/12/29 15:26 
# @File    : _shp_reader.py

import pathlib

import numpy as np
import geopandas as gpd
from rasterio import features
from affine import Affine
from shapely.geometry import Point, Polygon
from shapely.ops import transform
from shapely.prepared import prep
import shapely.vectorized
from pyproj import Transformer, CRS

from sagea import GRD
from sagea.utils import MathTool


class LoadShp:
    def __init__(self, filepath: pathlib.Path):
        self.__gpf = gpd.read_file(filepath)

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

    def __load_maskOLD(self, grid_space, mode="polygon", around_polar=False):
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

    def __load_mask(
            self,
            grid_space: int,
            per_feature: bool = True
    ):
        """
        Args:
          grid_space: Grid density; 1 means 1° grid
          per_feature: If True, return (n_feature, ny, nx), else return (ny, nx)
        """
        # 1) Read and reproject to lat/lon
        gdf = self.__gpf
        if gdf.crs is None:
            gdf.set_crs("EPSG:4326", inplace=True)

        gdf = gdf.to_crs("EPSG:4326")

        # 2) Build grid
        ny, nx = 180 * grid_space, 360 * grid_space
        d = 1.0 / grid_space

        # The upper-left pixel center should be (lat=89.5, lon=-179.5)
        # rasterio requires the upper-left pixel *corner* coordinate
        # Upper-left pixel center = (-179.5, 89.5)
        # So upper-left pixel corner = (-180, 90)
        transform = Affine(d, 0, -180, 0, -d, 90)
        lat, lon = MathTool.get_global_lat_lon_range(1)

        # 3) Rasterize
        if per_feature:
            masks = []
            for geom in gdf.geometry:
                mask = features.rasterize(
                    [(geom, 1)],
                    out_shape=(ny, nx),
                    transform=transform,
                    fill=0,
                    dtype="uint8",
                    all_touched=False  # only count pixels whose center is inside polygon
                )
                masks.append(mask)
            return np.stack(masks, axis=0)[:, ::-1, :], lat, lon  # (n_feature, ny, nx)

        else:
            mask = features.rasterize(
                [(geom, 1) for geom in gdf.geometry],
                out_shape=(ny, nx),
                transform=transform,
                fill=0,
                dtype="uint8",
                all_touched=False
            )
            return mask[::-1, :], lat, lon

    def __multipoint_to_one_mask(self, grid_space):
        pass

    def __load_bound(self):
        gdf = self.__gpf
        bound = np.array([gdf.bounds.minx, gdf.bounds.maxx, gdf.bounds.miny, gdf.bounds.maxy]).T

        return bound

    def get_GRD(self, grid_space, per_feature: bool = True):
        mask, lat, lon = self.__load_mask(grid_space, per_feature=per_feature)

        return GRD(mask, lat, lon)

    def get_SHC(self, lmax: int, spatial_accuracy: int | None = None, per_feature: bool = True):
        if spatial_accuracy is not None:
            grid_space = spatial_accuracy
        else:
            grid_space = int(180 / lmax)

        grid = self.get_GRD(grid_space, per_feature=per_feature)
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


def read_shp_as_GRD(filepath, grid_space, per_feature=True):
    load = LoadShp(filepath)

    return load.get_GRD(grid_space, per_feature=per_feature)


def read_shp_as_SHC(filepath, lmax, per_feature=True):
    load = LoadShp(filepath)
    return load.get_SHC(lmax, per_feature=per_feature)
