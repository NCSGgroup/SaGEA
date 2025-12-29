#!/use/bin/env python
# coding=utf-8
# @Author  : Shuhao Liu
# @Time    : 2025/12/29 15:06 
# @File    : core.py
import numpy
import numpy as np
from pathlib import Path
from src.sagea.io.gfc_reader import read_gfc
from src.sagea.utils.MathTool import MathTool


class SHC:
    """Spherical Harmonic coefficients"""

    # --- generate SHC instance --- #
    # Following methods will generate and return SHC instance classes
    def __init__(self, cs: numpy.ndarray, normalized="4pi"):
        assert cs.ndim in (1, 2)
        assert normalized in ("4pi",)

        self.value = cs if cs.ndim == 2 else cs[None, :]
        lmax_approx = np.sqrt(self.value.shape[1]) - 1
        assert np.isclose(lmax_approx, np.round(lmax_approx), atol=1e-8), "Invalid shape for CS arrays"

        self.normalized = normalized

    @staticmethod
    def from_file(filepath: Path or list[Path], lmax, key="gfc", cols=None, normalized="4pi"):
        """
        load SHC from filepath or list[filepath]
        """
        assert isinstance(filepath, Path) or isinstance(filepath, list)

        if isinstance(filepath, Path):
            cs_array = read_gfc(filepath, key=key, lmax=lmax, col_indices=cols)
        elif isinstance(filepath, list):
            cs_array = []
            for path in filepath:
                cs = read_gfc(path, key=key, lmax=lmax, col_indices=cols)
                cs_array.append(cs)
            cs_array = numpy.array(cs_array)
        else:
            assert False

        return SHC(cs_array, normalized=normalized)

    # --- properties and statistics --- #
    # Following methods will return required information
    @property
    def lmax(self):
        return int(np.sqrt(self.value.shape[1]) - 1)

    def __len__(self):
        return self.value.shape[0]

    @property
    def cs2d(self):
        """
        tuple:
            cqlm, sqlm.
            Both cqlm and sqlm are 3-dimension, EVEN IF self.__len__() > 1
        """
        lmax = self.lmax
        num_of_series = np.shape(self.value)[0]

        cqlm = np.zeros((num_of_series, lmax + 1, lmax + 1))
        sqlm = np.zeros((num_of_series, lmax + 1, lmax + 1))

        for i in range(num_of_series):
            this_cs = self.value[i]
            this_clm, this_slm = MathTool.cs_decompose_triangle1d_to_cs2d(this_cs)
            cqlm[i, :, :] = this_clm
            sqlm[i, :, :] = this_slm

        return cqlm, sqlm

    @property
    def mean(self):
        return np.mean(self.value, axis=0)

    @property
    def get_std(self):
        return np.std(self.value, axis=0)

    @property
    def get_var(self):
        return np.cov(self.value.T)

    @property
    def degree_rms(self):
        cqlm, sqlm = self.cs2d
        return MathTool.get_degree_rms(cqlm, sqlm)

    @property
    def degree_rss(self):
        cqlm, sqlm = self.cs2d
        return MathTool.get_degree_rss(cqlm, sqlm)

    @property
    def cumulative_degree_rss(self):
        cqlm, sqlm = self.cs2d
        return MathTool.get_cumulative_rss(cqlm, sqlm)

    # --- calculation and processing --- #
    # Following methods process and change the values in the instance
    def __add__(self, other):
        assert isinstance(other, SHC)
        assert self.lmax == other.lmax

        return SHC(self.value + other.value)

    def __sub__(self, other):
        assert isinstance(other, SHC)
        assert self.lmax == other.lmax

        return SHC(self.value - other.value)

    def de_mean(self):
        self.value -= self.mean

        return self

    def convert_type(self, from_type=None, to_type=None):
        types = list(enums.PhysicalDimensions)
        types_string = [i.name.lower() for i in types]
        types += types_string

        if from_type is None:
            from_type = Enums.PhysicalDimensions.Dimensionless
        if to_type is None:
            to_type = Enums.PhysicalDimensions.Dimensionless

        assert (from_type.lower() if type(
            from_type) is str else from_type) in types, f"from_type must be one of {types}"
        assert (to_type.lower() if type(
            to_type) is str else to_type) in types, f"to_type must be one of {types}"

        if type(from_type) is str:
            from_type = match_string(from_type, Enums.PhysicalDimensions, ignore_case=True)
        if type(to_type) is str:
            to_type = match_string(to_type, Enums.PhysicalDimensions, ignore_case=True)

        lmax = self.get_lmax()
        LN = LoveNumber()
        LN.configuration.set_lmax(lmax)

        ln = LN.get_Love_number()

        convert = ConvertSHC()
        convert.configuration.set_Love_number(ln).set_input_type(from_type).set_output_type(to_type)

        self.value = convert.apply_to(self.value)
        return self


if __name__ == '__main__':
    cs = np.random.normal(0, 1, size=(300, 3721,))

    shc = SHC(cs)
    shc.de_mean()
