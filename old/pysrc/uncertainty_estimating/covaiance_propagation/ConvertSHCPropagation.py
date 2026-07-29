import numpy as np

from pysrc.auxiliary.aux_tool.MathTool import MathTool
from pysrc.post_processing.convert_field_physical_quantity.ConvertSHC import ConvertSHC


class ConvertSHCPropagation(ConvertSHC):
    """
    Propagation of covariance information in the process of Gaussian filtering.
    """

    def __init__(self):
        super().__init__()

    def apply_to(self, cov_cs):
        """
        :param cov_cs: 2d-array, covariance matrix of the spherical harmonic coefficients which is sorted by degree,
                    for example (sgm stands for sigma),
     [[    var(c(0,0))    , sgm(c(0,0), c(1,0)), sgm(c(0,0), c(1,1)), sgm(c(0,0), s(1,1)), sgm(c(0,0), c(2,0)), ...],
      [sgm(c(1,0), c(0,0)),     var(c(1,0))    , sgm(c(1,0), c(1,1)), sgm(c(1,0), s(1,1)), sgm(c(1,0), c(2,0)), ...],
      [sgm(c(1,1), c(0,0)), sgm(c(1,1), c(1,0)),     var(c(1,1))    , sgm(c(1,1), s(1,1)), sgm(c(1,1), c(2,0)), ...],
      [sgm(s(1,0), c(0,0)), sgm(s(1,0), c(1,0)), sgm(s(1,0), c(1,1)),     var(s(1,1))    , sgm(s(1,1), c(2,0)), ...],
      [sgm(c(2,0), c(0,0)), sgm(c(2,0), c(1,0)), sgm(c(2,0), c(1,1)), sgm(c(2,0), s(1,1)),     var(c(2,0))    , ...],
      [                  :,                   :,                   :,                   :,                   :, ...]]
        """
        length_cs = np.shape(cov_cs)[0]
        lmax = int(np.sqrt(length_cs) - 1)

        convert_array_by_degree = self._get_convert_array_to_dimensionless(
            lmax) * self._get_convert_array_from_dimensionless_to(
            self.configuration.output_field_type, lmax)
        # [k0, k1, k2, ..., kn]

        convert_weight_cs1d = np.array([])
        for i in range(lmax + 1):
            convert_weight_cs1d = np.concatenate([convert_weight_cs1d, [convert_array_by_degree[i]] * (2 * i + 1)])

        convert_mat = np.diag(convert_weight_cs1d)

        # convert_mat = np.tile(convert_array_by_degree, (lmax + 1, 1)).T
        # convert_mat = MathTool.cs_2dto1d(convert_mat, MathTool.CS1dSortedBy.order)
        # convert_mat = np.diag(np.concatenate([convert_mat, convert_mat]))

        # sorted by order, [k(c00), k(c10), ..., k(s00), k(s10), ...]

        return convert_mat @ cov_cs @ convert_mat.T
