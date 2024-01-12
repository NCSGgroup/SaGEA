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

    def synthesis_mat_for_csqlm(self, cqlm: iter, sqlm: iter, Nmax, lat, lon, Pnm):
        assert len(cqlm) == len(sqlm)

        gqij = []
        for i in range(len(cqlm)):
            gqij.append(self.synthesisOld(cqlm[i], sqlm[i], Nmax, lat, lon, Pnm))

        return np.array(gqij)

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

        # Am, Bm = np.zeros((Nmax + 1, nlat)), np.zeros((Nmax + 1, nlat))
        Am, Bm = np.zeros((Nmax + 1, nlat)), np.zeros((Nmax + 1, nlat))
        I_new = Inner.reshape(-1, nlon)

        # for m in range(Nmax + 1):
        #     Am[m] = factor1 * np.array(I_new * np.mat(np.cos(m * phi)).T).flatten()
        #     Bm[m] = factor1 * np.array(I_new * np.mat(np.sin(m * phi)).T).flatten()

        for m in range(Nmax + 1):
            for i in range(nlat):
                # Am[m] = factor1 * np.array(I_new * np.mat(np.cos(m * phi)).T).flatten()
                Am[m, i] = factor1 * np.sum(I_new[i] * np.cos(m * phi))
                # Bm[m] = factor1 * np.array(I_new * np.mat(np.sin(m * phi)).T).flatten()
                Bm[m, i] = factor1 * np.sum(I_new[i] * np.sin(m * phi))

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

    def synthesis_non_mat(self, clm, slm):
        f = np.zeros((self.nlat, self.nlon))

        for i in trange(self.nlat):
            plm = self.pilm[i]

            for j in range(self.nlon):
                at_lon = self.lon[j]

                f[i, j] = 0
                for l in range(self.lmax + 1):
                    for m in range(l + 1):
                        f[i, j] += plm[l, m] * (clm[l, m] * np.cos(m * at_lon) + slm[l, m] * np.sin(m * at_lon))

        return f

    def analysis_non_mat(self, f):
        clm = np.zeros((self.lmax + 1, self.lmax + 1))
        slm = np.zeros((self.lmax + 1, self.lmax + 1))

        for l in trange(self.lmax + 1):

            for m in range(l + 1):

                for i in range(self.nlat):
                    at_lat = self.lat[i]
                    plm = self.pilm[i]

                    for j in range(self.nlon):
                        at_lon = self.lon[j]

                        clm[l, m] += f[i, j] * plm[l, m] * np.cos(m * at_lon) * np.sin(at_lat) * 2 * np.pi / (
                                self.nlon * self.nlat * 4)

                        slm[l, m] += f[i, j] * plm[l, m] * np.sin(m * at_lon) * np.sin(at_lat) * 2 * np.pi / (
                                self.nlon * self.nlat * 4)

        return clm, slm


def demo1():
    """synthesis/analysis mat"""

    from pysrc.auxiliary.load_file.LoadL2SH import load_SH_simple
    from pysrc.auxiliary.tools.FileTool import FileTool

    '''load shc'''
    lmax = 60
    spatial_resolution = 1

    clm, slm = load_SH_simple(
        FileTool.get_project_dir('data/auxiliary/GIF48.gfc'),
        key='gfc',
        lmax=lmax,
        lmcs_in_queue=(2, 3, 4, 5)
    )

    lat, lon = MathTool.get_global_lat_lon_range(spatial_resolution)
    har = Harmonic(lat, lon, lmax=lmax, option=1)

    PnmMat = MathTool.get_Legendre(lat, lmax, option=1)
    Pnm = np.array([MathTool.cs_2dto1d(PnmMat[i]) for i in range(np.shape(PnmMat)[0])]).T

    max_length = 240
    time_costs_s = []
    time_costs_a = []
    for length in trange(1, max_length + 1, 5):

        this_cost_s = 0
        this_cost_a = 0
        for i in range(length):
            time5 = time.time()
            f = har.synthesisOld(clm, slm, lmax, lat, lon, Pnm)
            time6 = time.time()  # synthesis old

            time7 = time.time()
            c, s = har.analysisOld(f, Pnm)
            time8 = time.time()  # analysis old

            this_cost_s += time6 - time5
            this_cost_a += time8 - time7

        time_costs_s.append(this_cost_s)
        time_costs_a.append(this_cost_a)

    return list(range(1, max_length + 1)), time_costs_s, time_costs_a  # s mat, a mat


def demo2():
    """synthesis/analysis vec"""

    from pysrc.auxiliary.load_file.LoadL2SH import load_SH_simple
    from pysrc.auxiliary.tools.FileTool import FileTool

    '''load shc'''
    lmax = 60
    spatial_resolution = 1

    clm, slm = load_SH_simple(
        FileTool.get_project_dir('data/auxiliary/GIF48.gfc'),
        key='gfc',
        lmax=lmax,
        lmcs_in_queue=(2, 3, 4, 5)
    )

    lat, lon = MathTool.get_global_lat_lon_range(spatial_resolution)
    har = Harmonic(lat, lon, lmax=lmax, option=1)

    PnmMat = MathTool.get_Legendre(lat, lmax, option=1)
    Pnm = np.array([MathTool.cs_2dto1d(PnmMat[i]) for i in range(np.shape(PnmMat)[0])]).T

    max_length = 240
    time_costs_s = []
    time_costs_a = []
    for length in trange(1, max_length + 1):
        cqlm = np.array([clm] * length)
        sqlm = np.array([slm] * length)

        time5 = time.time()
        f = har.synthesis_for_csqlm(cqlm, sqlm)
        time6 = time.time()  # synthesis vec

        time7 = time.time()
        c, s = har.analysis_for_gqij(f)
        time8 = time.time()  # analysis vec

        this_cost_s = time6 - time5
        this_cost_a = time8 - time7

        time_costs_s.append(this_cost_s)
        time_costs_a.append(this_cost_a)

    return list(range(1, max_length + 1)), time_costs_s, time_costs_a  # s mat, a mat


def demo3():
    """synthesis/analysis vec, CPU parallel"""

    from pysrc.auxiliary.load_file.LoadL2SH import load_SH_simple
    from pysrc.auxiliary.tools.FileTool import FileTool
    import multiprocessing as mp

    '''load shc'''
    lmax = 60
    spatial_resolution = 1

    clm, slm = load_SH_simple(
        FileTool.get_project_dir('data/auxiliary/GIF48.gfc'),
        key='gfc',
        lmax=lmax,
        lmcs_in_queue=(2, 3, 4, 5)
    )

    lat, lon = MathTool.get_global_lat_lon_range(spatial_resolution)
    har = Harmonic(lat, lon, lmax=lmax, option=1)

    PnmMat = MathTool.get_Legendre(lat, lmax, option=1)
    Pnm = np.array([MathTool.cs_2dto1d(PnmMat[i]) for i in range(np.shape(PnmMat)[0])]).T

    max_length = 240
    time_costs_s = []
    time_costs_a = []
    for length in trange(1, max_length + 1, 5):

        cqlm = np.array([clm] * length)
        sqlm = np.array([slm] * length)

        num_cores = int(mp.cpu_count())
        sets_each_group = length // num_cores + (length % num_cores > 0)

        params = [
            (cqlm[sets_each_group * i:sets_each_group * (i + 1)], sqlm[sets_each_group * i:sets_each_group * (i + 1)])
            for i in range(num_cores)
        ]

        time5 = time.time()
        har.synthesis_mat_for_csqlm(cqlm, sqlm, lmax, lat, lon, Pnm)
        time6 = time.time()  # synthesis vec

        time7 = time.time()
        for param in params:
            # p = mp.Process(target=har.synthesis_for_csqlm, args=param)
            p = mp.Process(target=har.synthesis_mat_for_csqlm, args=(*param, lmax, lat, lon, Pnm))
            p.start()
        time8 = time.time()  # analysis vec

        this_cost_s = time6 - time5
        this_cost_a = time8 - time7

        time_costs_s.append(this_cost_s)
        time_costs_a.append(this_cost_a)

    return list(range(1, max_length + 1)), time_costs_s, time_costs_a  # s mat, a mat


def demo4():
    """calculate time cost for several (average)"""

    repeat_times = 20

    time_cost = np.zeros(4)  # s new, a new, s old, a old
    for i in trange(repeat_times):
        this_time_cost = demo3()
        time_cost += np.array(this_time_cost)

    print('synthesis: before, synthesis: after, analysis: before, analysis: after')
    print(time_cost / repeat_times * 1000, 'ms')


def demo5():
    """synthesis/analysis for multi times"""

    from pysrc.auxiliary.load_file.LoadL2SH import load_SH_simple
    from pysrc.auxiliary.tools.FileTool import FileTool

    multi_times = 1
    lmax = 96
    spatial_resolution = 0.5

    '''load shc'''

    clm, slm = load_SH_simple(
        FileTool.get_project_dir('data/auxiliary/GIF48.gfc'),
        key='gfc',
        lmax=lmax,
        lmcs_in_queue=(2, 3, 4, 5)
    )

    lat, lon = MathTool.get_global_lat_lon_range(spatial_resolution)
    har = Harmonic(lat, lon, lmax=lmax, option=1)

    PnmMat = MathTool.get_Legendre(lat, lmax, option=1)

    time1 = time.time()
    # for i in range(multi_times):
    #     f = har.synthesis_non_mat(clm, slm)
    time2 = time.time()  # synthesis non-mat

    f = np.ones((180, 360))

    time3 = time.time()
    for i in range(multi_times):
        c, s = har.analysis_non_mat(f)
    time4 = time.time()  # analysis non-mat

    return np.array([time2 - time1, time4 - time3])  # s non-mat


def demo6():
    repeat_times = 5

    for i in range(repeat_times):
        this_time_cost = demo5()
        print(this_time_cost)


def demo7():
    # pass
    import pyshtools

    # degrees = np.arange(101, dtype=float)
    # degrees[0] = np.inf
    # power = degrees ** (-2)
    #
    # clm = pysh.SHCoeffs.from_random(power, seed=12345)

    # fig, ax = clm.plot_spectrum(show=False)

    # fig, ax = clm.plot_spectrum2d(cmap_rlimits=(1.e-7, 0.1),
    #                               show=False)

    # grid = clm.expand()
    # fig, ax = grid.plot(show=False)


if __name__ == '__main__':
    # x, ts, ta = demo1()
    # print(x, ts, ta)
    # print(ts, ta)
    demo7()