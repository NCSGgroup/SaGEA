from pathlib import Path

import numpy as np

from pysrc.data_class.GRD import GRD
from pysrc.post_processing.extract_basin_signal.ExtractSpatialSignalConfig import ExtractSpatialSignalConfig

from pysrc.auxiliary.load_file.LoadL2SH import load_SHC
from pysrc.auxiliary.preference.Constants import GeoConstants
from pysrc.post_processing.harmonic.Harmonic import Harmonic


class ExtractSpatial:
    def __init__(self):
        self.configuration = ExtractSpatialSignalConfig()

        self.basin = None
        self.signal = None

        self.weight = None

        self.radius_earth = GeoConstants.radius_earth

    def config(self, config: ExtractSpatialSignalConfig):
        self.configuration = config

        return self

    def set_basin(self, basin: np.ndarray or Path):
        """
        the input basin should have the same latitude and longitude range with those of self._configuration
        :param basin: 2d-array that describes a basin, with elements which equals to 1 inside the basin or 0 outside,
                        or Path that describes a filepath,
                        or class Basin
        """

        assert type(basin) in (np.ndarray,) or issubclass(type(basin), Path)

        if issubclass(type(basin), Path):
            basin = self.__load_SHC_to_basin(basin)

        basin_shape = np.shape(basin)
        assert basin_shape == (len(self.configuration.lat_range), len(self.configuration.lon_range))

        self.basin = basin
        return self

    def __load_SHC_to_basin(self, path: Path):
        lmax = 60
        clm_basin, slm_basin = load_SHC(path, key='', lmax=lmax, read_rows=(1, 2, 3, 4))
        har = Harmonic(self.configuration.lat_range, self.configuration.lon_range, lmax)
        grid_basin = har.synthesis(np.array([clm_basin]), np.array([slm_basin]))[0]

        return grid_basin

    def set_signal(self, grid: np.ndarray or GRD):
        """
        :param grid: 2d-array of gridded signal or 3d-array for series
        """
        if isinstance(type(grid), GRD):
            grid = grid.value

        if grid.ndim == 2:
            grid = np.array([grid])

        self.signal = grid
        return self

    def set_weight(self, grid: np.ndarray or GRD):
        if isinstance(type(grid), GRD):
            grid = grid.value

        if grid.ndim == 2:
            grid = np.array([grid])

        self.weight = grid
        return self

    def get_sum(self):
        """
        calculate the weighted sum of signals in the basin, using the area of each grid point as the weight.
        Note that when calculating the basin area,
        the difference in longitude/latitude between adjacent two points is used as the corresponding step size.
        Therefore,
        the gridded signal will be redistributed from shape(nlat, nlon) to shape(nlat-1, nlon-1) before calculation,
        using the average of adjacent two points.
        """
        signal_redistributed_pre = (self.signal[:, 1:, :] + self.signal[:, :-1, :]) / 2
        signal_redistributed = (signal_redistributed_pre[:, :, 1:] + signal_redistributed_pre[:, :, :-1]) / 2

        basin_redistributed_pre = (self.basin[1:, :] + self.basin[:-1, :]) / 2
        basin_redistributed = (basin_redistributed_pre[:, 1:] + basin_redistributed_pre[:, :-1]) / 2

        lon2d, lat2d = np.meshgrid(self.configuration.lon_range, self.configuration.lat_range)
        lat2d_redistributed_pre = (lat2d[1:, :] + lat2d[:-1, :]) / 2
        lat2d_redistributed = (lat2d_redistributed_pre[:, 1:] + lat2d_redistributed_pre[:, :-1]) / 2

        delta_lat = np.abs(self.configuration.lat_range[1:] - self.configuration.lat_range[:-1])
        delta_lon = np.abs(self.configuration.lon_range[1:] - self.configuration.lon_range[:-1])

        delta_lon2d, delta_lat2d = np.meshgrid(delta_lon, delta_lat)
        d_omega = self.radius_earth ** 2 * np.sin(lat2d_redistributed) * delta_lon2d * delta_lat2d

        return np.sum(signal_redistributed * basin_redistributed * d_omega)

    def get_average(self):
        """
        Calculate the weighted average of signals in the basin, using sine co-latitude of each grid point as the weight.
        """

        lon2d, lat2d = np.meshgrid(self.configuration.lon_range, self.configuration.lat_range)
        sin_theta = np.sin(lat2d)

        # if self.weight is None:
        #     weight = np.ones_like(self.signal)
        # else:
        #     weight = self.weight

        if self.weight is None:
            return np.sum(self.signal * sin_theta * self.basin, axis=(1, 2)) / np.sum(
                sin_theta * self.basin)
        else:
            # return np.sum(self.signal * sin_theta * self.weight * self.basin, axis=(1, 2)) / np.sum(
            #     sin_theta * self.weight * self.basin, axis=(1, 2))
            return np.sum(self.signal * sin_theta * self.weight * self.basin, axis=(1, 2)) / np.sum(
                sin_theta * self.weight * self.basin, axis=(1, 2))

    def get_area(self):
        """calculate area of basin"""
        basin_redistributed_pre = (self.basin[1:, :] + self.basin[:-1, :]) / 2
        basin_redistributed = (basin_redistributed_pre[:, 1:] + basin_redistributed_pre[:, :-1]) / 2

        lon2d, lat2d = np.meshgrid(self.configuration.lon_range, self.configuration.lat_range)
        lat2d_redistributed_pre = (lat2d[1:, :] + lat2d[:-1, :]) / 2
        lat2d_redistributed = (lat2d_redistributed_pre[:, 1:] + lat2d_redistributed_pre[:, :-1]) / 2

        delta_lat = np.abs(self.configuration.lat_range[1:] - self.configuration.lat_range[:-1])
        delta_lon = np.abs(self.configuration.lon_range[1:] - self.configuration.lon_range[:-1])

        delta_lon2d, delta_lat2d = np.meshgrid(delta_lon, delta_lat)
        d_omega = self.radius_earth ** 2 * np.sin(lat2d_redistributed) * delta_lon2d * delta_lat2d

        return np.sum(basin_redistributed * d_omega)


if __name__ == '__main__':
    from pysrc.auxiliary.aux_tool.FileTool import FileTool

    signal = np.random.uniform(0, 10, (180, 360))

    extra = ExtractSpatial()
    extra.set_basin(FileTool.get_project_dir() / 'data/basin_mask/Amazon_maskSH.dat')
    extra.set_signal(signal)

    ave = extra.get_average()
    sum_signal = extra.get_sum()
    area = extra.get_area()

    print('Greenland area (spatial)', area, (area - 5974818174868.082) / area)
