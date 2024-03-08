import numpy as np

from pysrc.auxiliary.aux_tool.MathTool import MathTool
from pysrc.post_processing.extract_basin_signal.ExtractSpectralSignal import ExtractSpectral


class BasinSumPropagation(ExtractSpectral):
    """
    Propagation of covariance information in the process of extracting basin sum/average signal.
    """

    def __init__(self):
        super().__init__()
        self.cov_cs = None

    def set_cov_mat(self, cov_cs):
        """
        :param cov_cs: 2d-array, covariance matrix of the spherical harmonic coefficients which is sorted by degree,
                    for example (sgm stands for sigma),
     [[    var(c(0,0))    , sgm(c(0,0), c(1,0)), sgm(c(0,0), c(1,1)), sgm(c(0,0), s(1,1)), sgm(c(0,0), c(2,0)), ...],
      [sgm(c(1,0), c(0,0)),     var(c(1,0))    , sgm(c(1,0), c(1,1)), sgm(c(1,0), s(1,1)), sgm(c(1,0), c(2,0)), ...],
      [sgm(c(1,1), c(0,0)), sgm(c(1,1), c(1,0)),     var(c(1,1))    , sgm(c(1,1), s(1,1)), sgm(c(1,1), c(2,0)), ...],
      [sgm(s(1,0), c(0,0)), sgm(s(1,0), c(1,0)), sgm(s(1,0), c(1,1)),     var(s(1,1))    , sgm(s(1,1), c(2,0)), ...],
      [sgm(c(2,0), c(0,0)), sgm(c(2,0), c(1,0)), sgm(c(2,0), c(1,1)), sgm(c(2,0), s(1,1)),     var(c(2,0))    , ...],
      [                  :,                   :,                   :,                   :,                   :, ...]]
        :return:
        """
        self.cov_cs = cov_cs
        return self

    def get_sum(self):
        """

        return: variance of basin TWS
        """
        basin_cs1d = MathTool.cs_combine_to_triangle_1d(self.basin_clm, self.basin_slm)

        return (self.radius_earth ** 2) ** 2 * basin_cs1d @ self.cov_cs @ basin_cs1d.T

    def get_average(self):
        """

        return: variance of basin EWH
        """
        return self.get_sum() / (self.get_area() ** 2)
