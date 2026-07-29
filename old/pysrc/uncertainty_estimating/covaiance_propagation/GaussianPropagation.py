import numpy as np

from pysrc.post_processing.filter.Gaussian import Gaussian


class GaussianPropagation(Gaussian):
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
        weight1d = self.get_weight_cs1d()
        gs_mat = np.diag(weight1d)
        return gs_mat @ cov_cs @ gs_mat.T
