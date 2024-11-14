import time

import numpy as np
from tqdm import trange

from pysrc.auxiliary.preference.EnumClasses import PhysicalDimensions
from pysrc.auxiliary.aux_tool.MathTool import MathTool

from pysrc.auxiliary.core_data_class.CoreGRID import CoreGRID
from pysrc.auxiliary.core_data_class.CoreSHC import CoreSHC


class Harmonic:
    """
    Harmonic analysis and synthesis: Ordinary 2D integration for computing Spherical Harmonic coefficients
    """

    def __init__(self, lat, lon, lmax: int, option=0):
        """

        :param lat: Co-latitude if option=0, unit[rad]; geophysical latitude if option = others, unit[degree]
        :param lon: If option=0, unit[rad]; else unit[degree]
        :param lmax: int, max degree/order
        :param option:
        """
        if option != 0:
            self.lat, self.lon = MathTool.get_colat_lon_rad(lat, lon)
        else:
            self.lat, self.lon = lat, lon

        self.lmax = lmax

        self._prepare()

    def _prepare(self):
        """pre-process some parameters of the 'two-step' method."""
        self.nlat, self.nlon = len(self.lat), len(self.lon)

        self.pilm = MathTool.get_Legendre(self.lat, self.lmax)
        # Associative Legendre polynomials, indexes stand for (co-lat[rad], degree l, order m)
        # 3-d array [theta, l, m] shape: (nlat * (lmax + 1) * (lmax + 1))

        # self.si = np.sin(self.lat) * np.pi / (2 * self.lmax + 1)

        # get Neumann weights
        # x_mat = np.array([np.cos(self.lat) ** i for i in range(self.nlat)])
        # r_vec = np.ones(self.nlat) * 2 / np.arange(1, self.nlat + 1, 1)
        # r_vec[np.arange(1, self.nlat + 1, 2)] = 0
        # self.wi = np.linalg.pinv(x_mat) @ r_vec

        m = np.arange(self.lmax + 1)
        self.g = m[:, None] @ self.lon[None, :]

        self.factor1 = np.ones((self.nlat, self.lmax + 1))
        self.factor1[:, 0] += 1
        self.factor1 = 1 / (self.factor1 * self.nlon)

        self.factor2 = np.ones((self.lmax + 1, self.lmax + 1))
        self.factor2[:, 0] += 1
        self.factor2 *= np.pi / (2 * self.nlat)

        self.factor3 = np.ones((self.lmax + 1, self.lmax + 1))
        self.factor3[:, 0] += 1
        self.factor3 /= 2
        pass

    # def analysis(self, grid: CoreGRID):
    #     """
    #     grid: data class GRID
    #     """
    #     # assert MathTool.get_colat_lon_rad(grid.lat, grid.lon) == (self.lat, self.lon)
    #     # colat_lon_of_grid = MathTool.get_colat_lon_rad(grid.lat, grid.lon)
    #     # assert all(abs(colat_lon_of_grid[0] - self.lat) < 1e-14) and all(abs(colat_lon_of_grid[1] - self.lon) < 1e-14)
    #
    #     if grid.is_series():
    #         gqij = grid.value
    #     else:
    #         gqij = np.array([grid.value])
    #
    #     cqlm, sqlm = self.analysis_for_gqij(gqij)
    #
    #     return CoreSHC(cqlm, sqlm)

    # def synthesis(self, shc: CoreSHC, special_type: PhysicalDimensions = None):
    #     assert shc.get_lmax() == self.lmax
    #     assert special_type in (
    #         PhysicalDimensions.HorizontalDisplacementEast, PhysicalDimensions.HorizontalDisplacementNorth, None)
    #
    #     cqlm, sqlm = shc.get_cs2d()
    #
    #     grids = self.synthesis_for_csqlm(cqlm, sqlm, special_type)
    #
    #     return CoreGRID(grids, self.lat, self.lon)

    def analysis_for_gqij(self, gqij: np.ndarray, special_type: PhysicalDimensions = None):
        assert len(gqij.shape) in (2, 3)

        single = (len(gqij.shape) == 2)
        if single:
            gqij = np.array([gqij])

        assert special_type in (
            PhysicalDimensions.HorizontalDisplacementEast, PhysicalDimensions.HorizontalDisplacementNorth, None)

        if special_type in (
                PhysicalDimensions.HorizontalDisplacementEast, PhysicalDimensions.HorizontalDisplacementNorth):
            assert False, "Horizontal Displacement is not supported yet."

        g = self.g
        co = np.cos(g)
        so = np.sin(g)

        am = np.einsum('pij,mj->pim', gqij, co, optimize='greedy') * self.factor1
        bm = np.einsum('pij,mj->pim', gqij, so, optimize='greedy') * self.factor1
        if special_type is None:
            cqlm = np.einsum('pim,ilm,i->plm', am, self.pilm, np.sin(self.lat), optimize='greedy') * self.factor2
            sqlm = np.einsum('pim,ilm,i->plm', bm, self.pilm, np.sin(self.lat), optimize='greedy') * self.factor2

            # cqlm = np.einsum('pim,ilm,i->plm', am, self.pilm, self.wi, optimize='greedy') * self.factor3
            # sqlm = np.einsum('pim,ilm,i->plm', bm, self.pilm, self.wi, optimize='greedy') * self.factor3

            # cqlm[:, :, 0] = (np.einsum('pim,ilm->plm', am * self.wi[:, None], self.pilm,
            #                               optimize='greedy') * self.factor3)[:, :, 0]
            # sqlm[:, :, 0] = (np.einsum('pim,ilm->plm', bm * self.wi[:, None], self.pilm,
            #                            optimize='greedy') * self.factor3)[:, :, 0]

            # cqlm = np.einsum('pim,ilm->plm', am * self.wi[:, None], self.pilm,
            #                  optimize='greedy') * self.factor3
            # sqlm = np.einsum('pim,ilm->plm', bm * self.wi[:, None], self.pilm,
            #                  optimize='greedy') * self.factor3
        elif special_type == PhysicalDimensions.HorizontalDisplacementNorth:
            pilm_derivative = MathTool.get_Legendre_derivative(self.lat, self.lmax)

            cqlm = np.einsum('pim,ilm,i->plm', -am, pilm_derivative, np.sin(self.lat),
                             optimize='greedy') * self.factor2
            sqlm = np.einsum('pim,ilm,i->plm', -bm, pilm_derivative, np.sin(self.lat),
                             optimize='greedy') * self.factor2

        elif special_type == PhysicalDimensions.HorizontalDisplacementEast:
            mrange = np.arange(self.lmax + 1)
            pilm_divide_sin_theta = self.pilm / np.sin(self.lat)[:, None, None]
            m_pilm_divide_sin_theta = np.einsum("m,ilm->ilm", mrange, pilm_divide_sin_theta)

            am_east = np.einsum('pij,mj->pim', gqij, -so, optimize='greedy') * self.factor1
            bm_east = np.einsum('pij,mj->pim', gqij, co, optimize='greedy') * self.factor1

            cqlm = np.einsum('pim,ilm,i->plm', am_east, m_pilm_divide_sin_theta, np.sin(self.lat),
                             optimize='greedy') * self.factor2
            sqlm = np.einsum('pim,ilm,i->plm', bm_east, m_pilm_divide_sin_theta, np.sin(self.lat),
                             optimize='greedy') * self.factor2

        else:
            assert False

        if single:
            assert cqlm.shape[0] == 1 and sqlm.shape[0] == 1
            return cqlm[0], sqlm[0]

        else:
            return cqlm, sqlm

    def synthesis_for_csqlm(self, cqlm: np.ndarray, sqlm: np.ndarray, special_type: PhysicalDimensions = None):
        assert cqlm.shape == sqlm.shape
        assert len(cqlm.shape) in (2, 3)
        single = (len(cqlm.shape) == 2)
        if single:
            cqlm = np.array([cqlm])
            sqlm = np.array([sqlm])

        assert special_type in (
            PhysicalDimensions.HorizontalDisplacementEast, PhysicalDimensions.HorizontalDisplacementNorth, None)

        cqlm = np.array(cqlm)
        sqlm = np.array(sqlm)

        if special_type is None:
            am = np.einsum('ijk,ljk->ilk', cqlm, self.pilm)
            bm = np.einsum('ijk,ljk->ilk', sqlm, self.pilm)

        elif special_type is PhysicalDimensions.HorizontalDisplacementNorth:
            pilm_derivative = MathTool.get_Legendre_derivative(self.lat, self.lmax)

            am = np.einsum('ijk,ljk->ilk', -cqlm, pilm_derivative)
            bm = np.einsum('ijk,ljk->ilk', -sqlm, pilm_derivative)

        elif special_type is PhysicalDimensions.HorizontalDisplacementEast:
            pilm_divide_sin_theta = self.pilm / np.sin(self.lat)[:, None, None]
            mrange = np.arange(self.lmax + 1)

            am = np.einsum('ijk,k,ljk->ilk', sqlm, mrange, pilm_divide_sin_theta)
            bm = np.einsum('ijk,k,ljk->ilk', -cqlm, mrange, pilm_divide_sin_theta)

        else:
            assert False

        co = np.cos(self.g)
        so = np.sin(self.g)

        gqij = am @ co + bm @ so

        if single:
            assert gqij.shape[0] == 1
            return gqij[0]
        else:
            return gqij


def demo1():
    """synthesis/analysis for once"""

    from pysrc.auxiliary.load_file.LoadL2SH import load_SHC
    from pysrc.auxiliary.aux_tool.FileTool import FileTool

    '''load shc'''
    lmax = 96
    spatial_resolution = 0.5

    clm, slm = load_SHC(
        FileTool.get_project_dir('data/auxiliary/GIF48.gfc'),
        key='gfc',
        lmax=lmax,
        lmcs_in_queue=(2, 3, 4, 5)
    )
    cqlm, sqlm = np.array([clm]), np.array([slm])

    lat, lon = MathTool.get_global_lat_lon_range(spatial_resolution)
    har = Harmonic(lat, lon, lmax=lmax, option=1)

    time1 = time.time()
    f = har.synthesis_for_csqlm(cqlm, sqlm)
    time2 = time.time()  # synthesis new

    time3 = time.time()
    c, s = har.analysis_for_gqij(f)
    time4 = time.time()  # analysis new

    # ----------------------------------

    PnmMat = MathTool.get_Legendre(lat, lmax, option=1)
    Pnm = np.array([MathTool.cs_2dto1d(PnmMat[i]) for i in range(np.shape(PnmMat)[0])]).T

    time5 = time.time()
    f = har.synthesisOld(clm, slm, lmax, lat, lon, Pnm)
    time6 = time.time()  # synthesis old

    time7 = time.time()
    c, s = har.analysisOld(f, Pnm)
    time8 = time.time()  # analysis old

    return time6 - time5, time2 - time1, time8 - time7, time4 - time3  # s old,s new,  a old, a new


def demo2():
    """calculate time cost for once (average)"""

    repeat_times = 1

    time_cost = np.zeros(4)  # s new, a new, s old, a old
    for i in trange(repeat_times):
        this_time_cost = demo1()
        time_cost += np.array(this_time_cost)

    print('synthesis: before, synthesis: after, analysis: before, analysis: after')
    print(time_cost / repeat_times * 1000, 'ms')


def demo3():
    """synthesis/analysis for multi times"""

    from pysrc.auxiliary.load_file.LoadL2SH import load_SHC
    from pysrc.auxiliary.aux_tool.FileTool import FileTool

    multi_times = 218
    lmax = 60
    spatial_resolution = 1

    '''load shc'''

    clm, slm = load_SHC(
        FileTool.get_project_dir('data/auxiliary/GIF48.gfc'),
        key='gfc',
        lmax=lmax,
        lmcs_in_queue=(2, 3, 4, 5)
    )
    cqlm, sqlm = np.array([clm] * multi_times), np.array([slm] * multi_times)

    lat, lon = MathTool.get_global_lat_lon_range(spatial_resolution)
    har = Harmonic(lat, lon, lmax=lmax, option=1)

    time1 = time.time()
    f = har.synthesis_for_csqlm(cqlm, sqlm)
    time2 = time.time()  # synthesis new

    time3 = time.time()
    cqlm, sqlm = har.analysis_for_gqij(f)
    time4 = time.time()  # analysis new

    # ----------------------------------

    PnmMat = MathTool.get_Legendre(lat, lmax, option=1)
    Pnm = np.array([MathTool.cs_2dto1d(PnmMat[i]) for i in range(np.shape(PnmMat)[0])]).T

    time5 = time.time()
    for i in range(multi_times):
        har.synthesisOld(clm, slm, lmax, lat, lon, Pnm)
    time6 = time.time()  # synthesis old

    time7 = time.time()
    for i in range(multi_times):
        har.analysisOld(np.array([f[i]]), Pnm)
    time8 = time.time()  # analysis old

    return np.array([time6 - time5, time2 - time1, time8 - time7, time4 - time3])  # s old, s new, a old, a new


def demo4():
    """calculate time cost for several (average)"""

    repeat_times = 20

    time_cost = np.zeros(4)  # s new, a new, s old, a old
    for i in trange(repeat_times):
        this_time_cost = demo3()
        time_cost += np.array(this_time_cost)

    print('synthesis: before, synthesis: after, analysis: before, analysis: after')
    print(time_cost / repeat_times * 1000, 'ms')


def demo_calculate_Neumann_weights():
    sp = 1
    lats = np.arange(-90. + sp / 2, 90. + sp / 2, sp)
    colats = np.radians(90. - lats)
    N = len(colats)

    cos_lats = np.cos(colats)
    x_mat = np.array([cos_lats ** i for i in range(N)])

    r_vec = np.ones(N) * 2 / np.arange(1, N + 1, 1)
    r_vec[np.arange(1, N + 1, 2)] = 0

    # print(x_mat)
    # print(r_vec)

    w = np.linalg.pinv(x_mat) @ r_vec
    print(w.shape)


def demo_fourier_mat():
    N = 60
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)

    pass


if __name__ == '__main__':
    # demo_calculate_Neumann_weights()
    demo_shtools()
