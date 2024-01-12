import time

import numpy as np
from tqdm import trange

from pysrc.auxiliary.preference.EnumClasses import FieldPhysicalQuantity
from pysrc.auxiliary.tools.MathTool import MathTool
from pysrc.auxiliary.core.GRID import GRID
from pysrc.auxiliary.core.SHC import SHC


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

        m = np.arange(self.lmax + 1)
        self.g = m[:, None] @ self.lon[None, :]

        self.factor1 = np.ones((self.nlat, self.lmax + 1))
        self.factor1[:, 0] += 1
        self.factor1 = 1 / (self.factor1 * self.nlon)

        self.factor2 = np.ones((self.lmax + 1, self.lmax + 1))
        self.factor2[:, 0] += 1
        self.factor2 *= np.pi / (2 * self.nlat)
        pass

    def analysis(self, grid: GRID):
        # assert MathTool.get_colat_lon_rad(grid.lat, grid.lon) == (self.lat, self.lon)
        colat_lon_of_grid = MathTool.get_colat_lon_rad(grid.lat, grid.lon)
        assert all(abs(colat_lon_of_grid[0] - self.lat) < 1e-14) and all(abs(colat_lon_of_grid[1] - self.lon) < 1e-14)

        if grid.is_series():
            gqij = grid.data
        else:
            gqij = np.array([grid.data])

        cqlm, sqlm = self.analysis_for_gqij(gqij)

        return SHC(cqlm, sqlm)

    def synthesis(self, shc: SHC, special_type: FieldPhysicalQuantity = None):
        assert shc.get_lmax() == self.lmax
        assert special_type in (
            FieldPhysicalQuantity.HorizontalDisplacementEast, FieldPhysicalQuantity.HorizontalDisplacementNorth, None)

        cqlm, sqlm = shc.get_cs2d()

        grids = self.synthesis_for_csqlm(cqlm, sqlm, special_type)

        return GRID(grids, self.lat, self.lon)

    def analysis_for_gqij(self, gqij: np.ndarray):
        g = self.g.T
        co = np.cos(g)
        so = np.sin(g)

        am = np.einsum('pij,jm->pim', gqij, co, optimize='greedy') * self.factor1
        bm = np.einsum('pij,jm->pim', gqij, so, optimize='greedy') * self.factor1

        cqlm = np.einsum('pim,ilm,i->plm', am, self.pilm, np.sin(self.lat), optimize='greedy') * self.factor2
        sqlm = np.einsum('pim,ilm,i->plm', bm, self.pilm, np.sin(self.lat), optimize='greedy') * self.factor2
        return cqlm, sqlm

    def synthesis_for_csqlm(self, cqlm: iter, sqlm: iter, special_type: FieldPhysicalQuantity = None):
        assert len(cqlm) == len(sqlm)
        assert special_type in (
            FieldPhysicalQuantity.HorizontalDisplacementEast, FieldPhysicalQuantity.HorizontalDisplacementNorth, None)

        cqlm = np.array(cqlm)
        sqlm = np.array(sqlm)

        if special_type is None:
            am = np.einsum('ijk,ljk->ilk', cqlm, self.pilm)
            bm = np.einsum('ijk,ljk->ilk', sqlm, self.pilm)

        elif special_type is FieldPhysicalQuantity.HorizontalDisplacementNorth:
            pilm_derivative = MathTool.get_Legendre_derivative(self.lat, self.lmax)

            am = np.einsum('ijk,ljk->ilk', sqlm, pilm_derivative)
            bm = np.einsum('ijk,ljk->ilk', -cqlm, pilm_derivative)

        elif special_type is FieldPhysicalQuantity.HorizontalDisplacementEast:
            pilm_divide_sin_theta = self.pilm / np.sin(self.lat)[:, None, None]

            am = np.einsum('ijk,ljk->ilk', -sqlm, pilm_divide_sin_theta)
            bm = np.einsum('ijk,ljk->ilk', -cqlm, pilm_divide_sin_theta)

        else:
            assert False

        co = np.cos(self.g)
        so = np.sin(self.g)

        gqij = am @ co + bm @ so

        return gqij

    def analysisOld(self, Inner, Pnm):
        Nmax = self.lmax
        lat, lon = self.lat, self.lon

        nlat = len(lat)
        nlon = len(lon)
        theta, phi = MathTool.get_colat_lon_rad(lat, lon)

        term1 = np.zeros(Nmax + 1)

        for l in range(0, Nmax + 1):
            term1[l] = 1 + 2 * l

        NMmax = int((Nmax + 1) * (Nmax + 2) / 2)

        factor1 = 2 * np.pi / nlon
        factor2 = 0.25 / nlat
        Cnm, Snm = np.zeros(NMmax), np.zeros(NMmax)

        Am, Bm = np.zeros((Nmax + 1, nlat)), np.zeros((Nmax + 1, nlat))
        I_new = Inner.reshape(-1, nlon)

        for m in range(Nmax + 1):
            Am[m] = factor1 * np.array(I_new * np.mat(np.cos(m * phi)).T).flatten()
            Bm[m] = factor1 * np.array(I_new * np.mat(np.sin(m * phi)).T).flatten()

        thetaS = np.tile(np.sin(theta), (MathTool.getIndex(Nmax, Nmax) + 1, 1))
        Qnm = Pnm * thetaS

        for n in range(Nmax + 1):
            indexM = np.arange(n + 1)
            Cnm[MathTool.getIndex(n, 0):MathTool.getIndex(n, n) + 1] = factor2 * np.sum(
                Qnm[MathTool.getIndex(n, 0):MathTool.getIndex(n, n) + 1] * Am[indexM], 1)
            Snm[MathTool.getIndex(n, 0):MathTool.getIndex(n, n) + 1] = factor2 * np.sum(
                Qnm[MathTool.getIndex(n, 0):MathTool.getIndex(n, n) + 1] * Bm[indexM], 1)
            pass
        # return Cnm, Snm
        return MathTool.cs_1dto2d(Cnm), MathTool.cs_1dto2d(Snm)

    def synthesisOld(self, Cnm, Snm, Nmax, lat, lon, Pnm):
        """
        A two step synthesis method, see the paper GJI (Nico Sneew)
        :param Cnm: in general, it should be the geo-potential coefficients sorted in 2 dimension.
        :param Snm:
        :param Nmax: Max degree of harmonic expansion.
        :param lat: geophysical latitude in unit "degree" [dimension : N]
        :param lon: geophysical latitude in unit "degree"[dimension : M]
        :return: grid (nlat*nlon) [dimension N*M]
        """

        lat, lon = MathTool.get_colat_lon_rad(lat, lon)
        nlat = np.size(lat)
        nlon = np.size(lon)
        Am, Bm = np.zeros((Nmax + 1, nlat)), np.zeros((Nmax + 1, nlat))
        for m in range(Nmax + 1):
            for l in range(m, Nmax + 1):
                index = MathTool.getIndex(l, m)
                Am[m] = Am[m] + Pnm[index] * Cnm[l][m]
                Bm[m] = Bm[m] + Pnm[index] * Snm[l][m]

        Fout = 0

        for m in range(Nmax + 1):
            co = np.cos(m * lon)
            so = np.sin(m * lon)
            Fout = Fout + np.mat(Am[m]).T * co + np.mat(Bm[m]).T * so

        return np.array(Fout)


def demo1():
    """synthesis/analysis for once"""

    from pysrc.auxiliary.load_file.LoadL2SH import load_SH_simple
    from pysrc.auxiliary.tools.FileTool import FileTool

    '''load shc'''
    lmax = 96
    spatial_resolution = 0.5

    clm, slm = load_SH_simple(
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

    from pysrc.auxiliary.load_file.LoadL2SH import load_SH_simple
    from pysrc.auxiliary.tools.FileTool import FileTool

    multi_times = 218
    lmax = 60
    spatial_resolution = 1

    '''load shc'''

    clm, slm = load_SH_simple(
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
    sp = 0.5
    lats = np.arange(-90. + sp / 2, 90. + sp / 2)
    colats = np.radians(90. - lats)
    N = len(colats)

    cos_lats = np.cos(colats)
    x_mat = np.array([cos_lats ** i for i in range(N)])

    r_vec = np.ones(N) * 2 / np.arange(1, N + 1, 1)
    r_vec[np.arange(1, N + 1, 2)] = 0

    # print(x_mat)
    # print(r_vec)

    w = np.linalg.pinv(x_mat) @ r_vec
    print(w)


if __name__ == '__main__':
    demo_calculate_Neumann_weights()
