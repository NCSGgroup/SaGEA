import pathlib
from pathlib import Path

import numpy as np

from pysrc.data_class.SHC import SHC
from pysrc.post_processing.extract_basin_signal.ExtractSpectralSignalConfig import ExtractSpectralSignalConfig

from pysrc.auxiliary.load_file.LoadL2SH import load_SHC
from pysrc.auxiliary.preference.Constants import GeoConstants


class ExtractSpectral:
    def __init__(self):
        self._configuration = ExtractSpectralSignalConfig()

        self.signal_cqlm = None  # 3d-array
        self.signal_sqlm = None

        self.basin_clm = None  # 2d-array
        self.basin_slm = None

        self.radius_earth = GeoConstants.radius_earth

    def config(self, config: ExtractSpectralSignalConfig):
        self._configuration = config

        return self

    def set_basin(self, *basin: np.ndarray or Path or SHC):
        """
        :param basin: two parameter clm and slm that describe the basin kernel function,
                        or one parameter Path that describes a filepath,
                        or class SHC
        """
        if len(basin) == 1:
            assert type(basin[0]) in (SHC,) or issubclass(type(basin[0]), Path)

            if issubclass(type(basin[0]), Path):
                path = basin[0]
                lmax = self._configuration.lmax
                self.basin_clm, self.basin_slm = load_SHC(path, key='', lmax=lmax, read_rows=(1, 2, 3, 4))

            elif type(basin[0]) is SHC:
                basin_cqlm, basin_sqlm = basin[0].get_cs2d()
                self.basin_clm, self.basin_slm = basin_cqlm[0], basin_sqlm[0]
        else:
            self.basin_clm = basin[0]
            self.basin_slm = basin[1]

        self.basin_clm *= 4 * np.pi
        self.basin_slm *= 4 * np.pi
        return self

    def set_signal(self, cqlm: np.ndarray, sqlm: np.ndarray):
        assert cqlm.ndim == sqlm.ndim
        if cqlm.ndim == 2:
            cqlm = np.array([cqlm])
            sqlm = np.array([sqlm])

        self.signal_cqlm = cqlm
        self.signal_sqlm = sqlm
        return self

    def get_sum(self):
        tot = self.signal_cqlm * self.basin_clm + self.signal_sqlm * self.basin_slm
        if np.ndim(self.signal_cqlm) == 3:
            index = 1
        else:
            index = 0
        cs_sum = np.sum(np.sum(tot, axis=index), axis=index)
        return self.radius_earth ** 2 * cs_sum

    def get_average(self):
        """
        Calculate the weighted average of signals in the basin, using sine co-latitude of each grid point as the weight.
        """
        return self.get_sum() / self.get_area()

    def get_area(self):
        """calculate area of basin"""

        return self.basin_clm[0, 0] * self.radius_earth ** 2


if __name__ == '__main__':
    from pysrc.auxiliary.aux_tool.FileTool import FileTool

    signal_c = np.random.normal(0, 100, (61, 61))
    signal_s = np.random.normal(0, 100, (61, 61))
    # signal = np.ones((180, 360))

    extra = ExtractSpectral()
    extra.set_basin(FileTool.get_project_dir() / 'data/basin_mask/Amazon_maskSH.dat')
    extra.set_signal(signal_c, signal_s)

    ave = extra.get_average()
    sum_signal = extra.get_sum()
    area = extra.get_area()

    print('Greenland area (spectral)', area)
