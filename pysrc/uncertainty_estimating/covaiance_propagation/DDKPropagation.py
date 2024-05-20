import numpy as np

from pysrc.data_class.DataClass import SHC
from pysrc.post_processing.filter.DDK import DDK


class DDKPropagation(DDK):
    """
    Propagation of covariance information in the process of Gaussian filtering.
    """

    def __init__(self):
        super().__init__()

    def get_ddk_mat(self, lmax):
        def get_shc_unit():
            cs_length = (lmax + 1) ** 2
            cs_unit = np.eye(cs_length)

            shc = SHC(cs_unit[0])

            print('generating unit SHC...')
            for i in range(1, np.shape(cs_unit)[0]):
                print(f'\r{i}/{np.shape(cs_unit)[0] - 1}', end='')
                shc.append(SHC(cs_unit[i]))

            print('\ndone!')

            return shc

        shc_unit = get_shc_unit()
        shc_filtered = self.apply_to(shc_unit)

        return shc_filtered.value

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
        lmax = int(np.sqrt(np.shape(cov_cs)[0]) - 1)
        ddk_mat = self.get_ddk_mat(lmax)
        return ddk_mat @ cov_cs @ ddk_mat.T
