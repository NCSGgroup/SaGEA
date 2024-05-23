import numpy as np
from tqdm import trange

from pysrc.auxiliary.aux_tool.MathTool import MathTool
from pysrc.post_processing.harmonic.Harmonic import Harmonic


class HarmonicPropagation(Harmonic):
    """
    Propagation of covariance information in the process of spherical harmonic synthesis or analysis.
    """

    def __init__(self, lat, lon, lmax: int, option=0):
        """

        :param lat: Co-latitude if option=0, unit[rad]; geophysical latitude if option = others, unit[degree]
        :param lon: If option=0, unit[rad]; else unit[degree]
        :param lmax:
        :param option:
        """
        super().__init__(lat, lon, lmax, option)
        self.synthesis_mat = None

    def synthesis_cov(self, cov_cs):
        """

        :param cov_cs: 2d-array, covariance matrix of the spherical harmonic coefficients which is sorted by degree,
                    for example (sgm stands for sigma),
     [[    var(c(0,0))    , sgm(c(0,0), c(1,0)), sgm(c(0,0), c(1,1)), sgm(c(0,0), s(1,1)), sgm(c(0,0), c(2,0)), ...],
      [sgm(c(1,0), c(0,0)),     var(c(1,0))    , sgm(c(1,0), c(1,1)), sgm(c(1,0), s(1,1)), sgm(c(1,0), c(2,0)), ...],
      [sgm(c(1,1), c(0,0)), sgm(c(1,1), c(1,0)),     var(c(1,1))    , sgm(c(1,1), s(1,1)), sgm(c(1,1), c(2,0)), ...],
      [sgm(s(1,0), c(0,0)), sgm(s(1,0), c(1,0)), sgm(s(1,0), c(1,1)),     var(s(1,1))    , sgm(s(1,1), c(2,0)), ...],
      [sgm(c(2,0), c(0,0)), sgm(c(2,0), c(1,0)), sgm(c(2,0), c(1,1)), sgm(c(2,0), s(1,1)),     var(c(2,0))    , ...],
      [                  :,                   :,                   :,                   :,                   :, ...]]

        :return: 2d-array, covariance matrix of the gridded data which is sorted by (co-)latitude theta,
                    for example, (th stands for theta, ph stands for phi)
                    [[sgm((th1, ph1), (th1, ph1)), sgm((th1, ph1), (th1, ph2)), ..., sgm((th1, ph1), (th2, ph1)), ...],
                     [sgm((th1, ph2), (th1, ph1)), sgm((th1, ph2), (th1, ph2)), ..., sgm((th1, ph2), (th2, ph1)), ...],
                     [                          :,                           :, ...,                           :, ...],
                     [sgm((th2, ph1), (th1, ph1)), sgm((th2, ph1), (th1, ph2)), ..., sgm((th2, ph1), (th2, ph1)), ...],
                     [                          :,                           :, ...,                           :, ...]]

        """
        hs_mat = self.get_synthesis_propagation_matrix()

        print('Calculating gridded covariance matrix...')
        '''cov_grid = A X AT'''
        AX = np.einsum('ij,jk->ij', hs_mat, cov_cs)
        AXAT = np.einsum('ij,jk->ij', AX, hs_mat.T)
        return AXAT

    def synthesis_var(self, cov_cs):
        """

        :param cov_cs: 2d-array, covariance matrix of the spherical harmonic coefficients which is sorted by degree,
                    for example (sgm stands for sigma),
     [[    var(c(0,0))    , sgm(c(0,0), c(1,0)), sgm(c(0,0), c(1,1)), sgm(c(0,0), s(1,1)), sgm(c(0,0), c(2,0)), ...],
      [sgm(c(1,0), c(0,0)),     var(c(1,0))    , sgm(c(1,0), c(1,1)), sgm(c(1,0), s(1,1)), sgm(c(1,0), c(2,0)), ...],
      [sgm(c(1,1), c(0,0)), sgm(c(1,1), c(1,0)),     var(c(1,1))    , sgm(c(1,1), s(1,1)), sgm(c(1,1), c(2,0)), ...],
      [sgm(s(1,0), c(0,0)), sgm(s(1,0), c(1,0)), sgm(s(1,0), c(1,1)),     var(s(1,1))    , sgm(s(1,1), c(2,0)), ...],
      [sgm(c(2,0), c(0,0)), sgm(c(2,0), c(1,0)), sgm(c(2,0), c(1,1)), sgm(c(2,0), s(1,1)),     var(c(2,0))    , ...],
      [                  :,                   :,                   :,                   :,                   :, ...]]
                    Note that though elements related to s[l,0] is theoretically equivalent to 0，
                    the storage still contains them to avoid unnecessary errors.

        :return: 2d-array, gridded variance of each point,
                    for example, (th stands for theta, ph stands for phi)
                    [[var(th1, ph1), var((th1, ph2)), var((th1, ph3)), ..., var(th1, phj), ...],
                     [var(th2, ph1), var((th2, ph2)), var((th2, ph3)), ..., var(th2, phj), ...],
                     [var(th3, ph1), var((th3, ph2)), var((th3, ph3)), ..., var(th3, phj), ...],
                     [            :,               :,               :, ...,             :, ...],
                     [var(thi, ph1), var((thi, ph2)), var((thi, ph3)), ..., var(thi, phj), ...],
                     [             :,              :,               :, ...,             :, ...]]

        """
        hs_mat = self.get_synthesis_propagation_matrix()

        '''var_grid = diag(A X AT)'''
        ax = hs_mat @ cov_cs
        axat_diag = np.sum(ax * hs_mat, axis=1)

        return axat_diag.reshape((self.nlat, self.nlon))

    def get_synthesis_propagation_matrix(self):
        if self.synthesis_mat is None:

            lon2d_index, lat2d_index = np.meshgrid(range(self.nlon), range(self.nlat))
            lon2dto1d_index, lat2dto1d_index = lon2d_index.flatten(), lat2d_index.flatten()
            # Note that lat2dto1d_index (lon2dto1d_index) is the one-dimensional formed by expanding lat2d_index,
            # rather than lat_index itself.

            '''get harmonic synthesis matrix'''
            hs_mat = np.zeros((self.nlat * self.nlon, (self.lmax + 1) ** 2))
            for i in range(len(lat2dto1d_index)):
                lat_index = lat2dto1d_index[i]
                # lon_index = lat2dto1d_index[i]
                lon_index = lon2dto1d_index[i]

                part_Legendre_2d = self.pilm[lat_index, :, :]
                Legendre_tri = np.concatenate([part_Legendre_2d[:, -1:0:-1], part_Legendre_2d], axis=1)

                part_cos_mphi_2d = np.tile(np.cos(np.arange(self.lmax + 1) * self.lon[lon_index]), (self.lmax + 1, 1))
                part_cos_mphi_tril = np.tril(part_cos_mphi_2d)

                part_sin_mphi_2d = np.tile(np.sin(np.arange(self.lmax + 1) * self.lon[lon_index]), (self.lmax + 1, 1))
                part_sin_mphi_tril = np.tril(part_sin_mphi_2d)

                sincos_tri = np.concatenate([part_sin_mphi_tril[:, -1:0:-1], part_cos_mphi_tril], axis=1)  # /sin|cos\

                Legendre_sincos_tri = Legendre_tri * sincos_tri  # /slm|clm\

                ones_tril = np.tril(np.ones((self.lmax + 1, self.lmax + 1)))
                ones_tri = np.concatenate([ones_tril[:, -1:0:-1], ones_tril], axis=1)
                index_tri = np.where(ones_tri == 1)

                p_vec = Legendre_sincos_tri[index_tri]

                hs_mat[i, :] = p_vec

            self.synthesis_mat = hs_mat

            return hs_mat

        else:
            return self.synthesis_mat
