import numpy as np

from pysrc.auxiliary.aux_tool.MathTool import MathTool
from pysrc.post_processing.filter.Base import get_poly_func, SHCFilter


class PnMmConfig:
    def __init__(self):
        self.poly_n = 3
        self.start_m = 10

    def set_n(self, n):
        self.poly_n = n

        return self

    def set_m(self, m):
        self.start_m = m

        return self


class PnMm(SHCFilter):
    def __init__(self):
        self.configuration = PnMmConfig()

    def _apply_to_cqlm(self, cqlm: np.ndarray):
        lmax = np.shape(cqlm)[1] - 1
        fit_function = get_poly_func(self.configuration.poly_n)

        def get_fit_array(array):
            t = np.arange(np.shape(array)[0]) + 1
            A = MathTool.get_design_matrix(fit_function, t)
            fit_params = np.linalg.pinv(A) @ array
            t_expand = np.array([t ** p for p in range(self.configuration.poly_n + 1)]).T
            fit_array = t_expand @ fit_params

            return fit_array

        for m in range(self.configuration.start_m, lmax + 1, 1):
            l1 = np.arange(m, lmax + 1, 2)
            l2 = np.arange(m + 1, lmax + 1, 2)
            if len(l1) <= self.configuration.poly_n or len(l2) <= self.configuration.poly_n:
                continue

            this_array1 = cqlm[:, l1, m].T
            this_array2 = cqlm[:, l2, m].T

            fit_this_array1 = get_fit_array(this_array1)
            fit_this_array2 = get_fit_array(this_array2)

            this_array1 -= fit_this_array1
            this_array2 -= fit_this_array2
            cqlm[:, l1, m] = this_array1.T
            cqlm[:, l2, m] = this_array2.T

        return cqlm

    # def apply_to(self, shc: SHC):
    #     cqlm, sqlm = shc.get_cs2d()
    #
    #     length_of_cqlm = np.shape(cqlm)[0]
    #     csqlm = np.concatenate([cqlm, sqlm])
    #     csqlm = self._apply_to_cqlm(csqlm)
    #
    #     cqlm_filtered = csqlm[:length_of_cqlm]
    #     sqlm_filtered = csqlm[length_of_cqlm:]
    #     return SHC(cqlm_filtered, sqlm_filtered)

    def apply_to(self, cqlm, sqlm):

        cqlm, sqlm, single = self._cs_to_3d_array(cqlm, sqlm)

        length_of_cqlm = np.shape(cqlm)[0]
        csqlm = np.concatenate([cqlm, sqlm])
        csqlm = self._apply_to_cqlm(csqlm)

        cqlm_f = csqlm[:length_of_cqlm]
        sqlm_f = csqlm[length_of_cqlm:]

        if single:
            assert cqlm_f.shape[0] == sqlm_f.shape[0] == 1
            return cqlm_f[0], sqlm_f[0]
        else:
            return cqlm_f, sqlm_f
