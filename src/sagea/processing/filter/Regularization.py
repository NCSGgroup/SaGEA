import numpy as np

from pysrc.auxiliary.aux_tool.MathTool import MathTool
from pysrc.post_processing.filter.Base import SHCFilter


class RegularizationConfig:
    def __init__(self):
        """default setting"""
        self.__vcm_err = None
        self.__vcm_sig = None
        self.__alpha = None

    @property
    def vcm_err(self):
        return self.__vcm_err

    @vcm_err.setter
    def vcm_err(self, vcm):
        """
        set error variance-covariance matrix,
        ordered as:
            c00, s11, c10, c11, s22, s21, c20, c21, c22, s33, ...
        """
        self.__vcm_err = vcm

    @property
    def vcm_sig(self):
        return self.__vcm_sig

    @vcm_sig.setter
    def vcm_sig(self, vcm):
        """
        set signal variance-covariance matrix,
        ordered as:
            c00, s11, c10, c11, s22, s21, c20, c21, c22, s33, ...
        """
        self.__vcm_sig = vcm

    @property
    def alpha(self):
        return self.__alpha

    @alpha.setter
    def alpha(self, alpha):
        self.__alpha = alpha


class Regularization(SHCFilter):
    def __init__(self):
        self.configuration = RegularizationConfig()

    def config(self, config: RegularizationConfig):
        self.configuration = config
        return self

    def apply_to(self, cqlm, sqlm):
        cs1d = MathTool.cs_combine_to_triangle_1d(cqlm, sqlm)

        c_mat = self.configuration.vcm_err
        d_mat = self.configuration.vcm_sig
        alpha = self.configuration.alpha

        a_C_Dinv = alpha * np.linalg.solve(d_mat.T, c_mat.T).T
        eye = np.eye(c_mat.shape[0])

        cs1d_filtered = np.linalg.solve(eye + a_C_Dinv, cs1d.T).T

        cqlm_f, sqlm_f = MathTool.cs_decompose_triangle1d_to_cs2d(cs1d_filtered)

        return cqlm_f, sqlm_f
