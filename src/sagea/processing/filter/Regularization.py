import numpy as np

from pysrc.auxiliary.aux_tool.MathTool import MathTool
from pysrc.post_processing.filter.Base import SHCFilter


class RegularizationConfig:
    def __init__(self):
        """default setting"""
        self.vcm_err = None
        self.vcm_sig = None
        self.alpha = None

    @property
    def vcm_err(self):
        return self.vcm_err

    @vcm_err.setter
    def vcm_err(self, vcm_err):
        """
        set error variance-covariance matrix,
        ordered as:
            c00, s11, c10, c11, s22, s21, c20, c21, c22, s33, ...
        """
        self.vcm_err = vcm_err

    @property
    def vcm_sig(self):
        return self.vcm_sig

    @vcm_sig.setter
    def vcm_sig(self, vcm_sig):
        """
        set signal variance-covariance matrix,
        ordered as:
            c00, s11, c10, c11, s22, s21, c20, c21, c22, s33, ...
        """
        self.vcm_sig = vcm_sig

    @property
    def alpha(self):
        return self.alpha

    @alpha.setter
    def alpha(self, alpha):
        self.alpha = alpha


class Regularization(SHCFilter):
    def __init__(self):
        self.configuration = RegularizationConfig()

    def config(self, config: RegularizationConfig):
        self.configuration = config
        return self

    def apply_to(self, cqlm, sqlm):
        cs1d = MathTool.cs_combine_to_triangle_1d(cqlm, sqlm)

        c_mat = self.config.vcm_err
        d_mat = self.config.vcm_sig
        alpha = self.config.alpha

        c_inv = np.linalg.inv(c_mat)
        d_inv = np.linalg.inv(d_mat)

        filtering_mat = np.linalg.inv(c_inv + alpha * d_inv) @ c_inv

        cs1d_filtered = (filtering_mat @ cs1d.T).T

        cqlm_f, sqlm_f = MathTool.cs_decompose_triangle1d_to_cs2d(cs1d_filtered)

        return cqlm_f, sqlm_f
