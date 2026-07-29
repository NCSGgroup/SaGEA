import numpy as np

from pysrc.auxiliary.aux_tool.MathTool import MathTool


class ExtractSpatialSignalConfig:
    def __init__(self):
        lat_range = np.arange(-89.5, 90.5, 1)
        lon_range = np.arange(-179.5, 180.5, 1)

        self.lat_range, self.lon_range = MathTool.get_colat_lon_rad(lat_range, lon_range)

    def set_lat_lon_range(self, lat, lon, option=1):
        """
        :param lat: geographic latitude in [deg] if option = 1 or co-latitude in [rad]
        :param lon: geographic longitude in [deg] if option = 1 or longitude in [rad]
        :param option: 1 or 0
        """
        assert option in (1, 0)
        if option == 1:
            lat, lon = MathTool.get_colat_lon_rad(lat, lon)

        self.lat_range = lat
        self.lon_range = lon

        return self
