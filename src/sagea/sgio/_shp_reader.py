#!/use/bin/env python
# coding=utf-8
# @Author  : Shuhao Liu
# @Time    : 2025/12/29 15:26 
# @File    : _shp_reader.py

import pathlib

import numpy as np

from sagea import GRD
from sagea.utils import MathTool


class LoadShp:
    def __init__(self, filepath: pathlib.Path):
        import geopandas as gpd
        self.__gpf = gpd.read_file(filepath)

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

        from rasterio import features
        from affine import Affine

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
