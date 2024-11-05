import numpy as np


class CoreGRID:
    """
    This class is to store the gridded signal for the use in necessary data processing.

    Attribute grid is the gridded signal, lat and lon are stored in unit [degree]
    """

    def __init__(self, grid, lat, lon, option=1):
        """
        To create a GRID object,
        one needs to specify the data (grid) and corresponding latitude range (lat) and longitude range (lon).
        :param grid: 2d- or 3d-array gridded signal, index ([num] ,lat, lon)
        :param lat: co-latitude in [rad] if option=0 else latitude in [degree]
        :param lon: longitude in [rad] if option=0 else longitude in [degree]
        :param option: 0 if colat and lon are in [rad]
        """
        if np.ndim(grid) == 2:
            grid = [grid]

        assert np.shape(grid)[-2:] == (len(lat), len(lon))

        self.value = np.array(grid)

        if option == 0:
            self.lat = 90 - np.degrees(lat)
            self.lon = np.degrees(lon)

        else:
            self.lat = lat
            self.lon = lon

        pass

    def append(self, grid, lat=None, lon=None, option=0):
        """

        :param grid: instantiated GRID or a 2d-array of index (lat, lon).
                        If 2d-array, the lat and lon range should be the same with self.lat and self.lon;
                        If instantiated GRID, params lat, lon and option are not needed.
        :param lat: co-latitude in [rad] if option=0 else latitude in [degree]
        :param lon: longitude in [rad] if option=0 else longitude in [degree]
        :param option:
        :return:
        """
        assert type(grid) in (CoreGRID, np.ndarray)

        if type(grid) is CoreGRID:
            assert lat is None and lon is None
            assert grid.lat == self.lat
            assert grid.lon == self.lon

        else:
            assert np.shape(grid)[-2:] == (len(self.lat), len(self.lon))
            grid = CoreGRID(grid, self.lat, self.lon, option)

        array_to_append = grid.value if grid.is_series() else np.array([grid.value])
        array_self = self.value if self.is_series() else [self.value]

        self.value = np.concatenate([array_self, array_to_append])

        return self

    def is_series(self):
        """
        To determine whether the data stored in this class are one group or multiple groups.
        :return: bool, True if it stores multiple groups, False if it stores only one group.
        """
        return len(np.shape(self.value)) == 3

    def get_grid_space(self):
        """
        return: grid_space in unit [degree]
        """
        return round(self.lat[1] - self.lat[0], 2)
