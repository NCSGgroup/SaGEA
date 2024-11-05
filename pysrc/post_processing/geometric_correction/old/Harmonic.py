"""
@Company: CGE-HUST, Wuhan, China
@Version: 1.0
@Author: Yang Fan
@Contact : yfan_cge@hust.edu.cn
@Modify Time: 2020/6/9 14:14
@Description:
"""
from multiprocessing import Pool

import numpy as np

from pysrc.post_processing.geometric_correction.old.GeoMathKit import GeoMathKit
from pysrc.post_processing.geometric_correction.old.LoveNumber import LoveNumber
from pysrc.post_processing.geometric_correction.old.RefEllipsoid import RefEllipsoid
from pysrc.post_processing.geometric_correction.old.Setting import SynthesisType, EllipsoidType, \
    LoveNumberType, Constants, HarAnalysisType


class Harmonic:
    """
    Harmonic analysis and synthesis: Ordinary 2D integration for computing Spherical Harmonic coefficients
    """

    def __init__(self, Ln: LoveNumber, Parallel: int = -1):
        self._Ln = Ln
        self._parallel = Parallel
        self._LoveMethod = LoveNumberType.AOD04
        self._ellipsoid = RefEllipsoid(EllipsoidType.gif48)
        pass

    def setLoveNumMethod(self, method: LoveNumberType):
        self._LoveMethod = method
        return self

    def setEllipsoid(self, ell: EllipsoidType):
        self._ellipsoid = RefEllipsoid(ell)
        return self

    @DeprecationWarning
    def analysis_old(self, Nmax: int, Inner: list, lat, lon, Pnm, kind=HarAnalysisType.InnerIntegral):
        """
        A simpson-type quadrature method.
        lat,lon, I and Pnm should keep accordance.
        The output is SH coefficients in terms of geo-potential.
        Notice: the ordering of Inner has to be N * M.  (rows sorted by latitude)

        Notice: the grid resolution has indeed determined the max degree (Nmax) of SH expansion, therefore the user
        defined Nmax can not be set without limitation, otherwise the high degree/order of output SH is fake.
        For regular grid that spans for N * 2N, the Nmax is suggested to be less than N/2-1.
        For Driscoll-healy grid that spans for N * N, the Nmax is suggested to be less than N/2-1.0
        For Gauss-Legendre quadrature grid that spans for (N+1)*(2*N+1), the Nmax is suggested to be less than N.

        :param kind: to let the function knows what type of the input gridded fields is
        :param Nmax: maximal degree of spherical harmonic expansion
        :param Inner: degree-dependent radial integration, len=Nmax or a simple gridded field len=1
                        dimension: [len, N*M]
        :param lat: latitude in degree   [dimension: N]
        :param lon: longitude in degree [dimension: M]
        :param Pnm: Legendre function
        :return: A set of spherical harmonic coefficients up to degree Nmax with one-dimension ordering type
        """
        # -------------------------------------------------------------
        # # extract the one-line lat and lon .
        # lat, lon = GeoMathKit.getCoLatLoninRad(lat, lon)
        #
        # # nlon = 2 * nlat
        # phi = lon[0:nlon]
        #
        # theta = np.zeros(nlat)
        # for i in range(nlat):
        #     theta[i] = lat[i * nlon]
        nlat = len(lat)
        nlon = len(lon)
        '''Co-latitude, longitude in [rad] '''
        theta, phi = GeoMathKit.getCoLatLoninRad(lat, lon)
        # --------------------------------------------------------------

        term1 = np.zeros(Nmax + 1)

        for l in range(0, Nmax + 1):
            term1[l] = 1 + 2 * l

        # kl = self._Ln.getNumber(Nmax, self._LoveMethod)
        # M = self._ellipsoid.Mass
        # R = self._ellipsoid.SemimajorAxis
        # factorSH = 4 * np.pi * R ** 2 / M * (1 + kl) / term1

        factorSH = self._factorHarAnalysis(Nmax=Nmax, kind=kind)

        # ---------------------------------------------------------------

        NMmax = int((Nmax + 1) * (Nmax + 2) / 2)

        factor1 = 2 * np.pi / nlon
        factor2 = 0.25 / nlat
        Cnm, Snm = np.zeros(NMmax), np.zeros(NMmax)

        if len(Inner) == 1:
            Am, Bm = np.zeros((Nmax + 1, nlat)), np.zeros((Nmax + 1, nlat))
            I_new = Inner[0].reshape(-1, nlon)

            for m in range(Nmax + 1):
                for i in range(nlat):
                    Am[m, i] = factor1 * np.sum(np.cos(m * phi) * I_new[i])
                    Bm[m, i] = factor1 * np.sum(np.sin(m * phi) * I_new[i])

            for n in range(Nmax + 1):
                for m in range(n + 1):
                    Cnm[GeoMathKit.getIndex(n, m)] = factorSH[n] * factor2 * np.sum(
                        Pnm[GeoMathKit.getIndex(n, m)] * np.sin(theta) * Am[m])
                    Snm[GeoMathKit.getIndex(n, m)] = factorSH[n] * factor2 * np.sum(
                        Pnm[GeoMathKit.getIndex(n, m)] * np.sin(theta) * Bm[m])

            return Cnm, Snm

        # -------------------------------------------------------------
        if len(Inner) > 1:
            assert len(Inner) >= (Nmax + 1)

        self.__input = (Nmax, nlat, nlon, Inner, factor1, phi, factorSH, Cnm, Snm, factor2, Pnm, theta)

        if self._parallel > 1:
            # for multiprocessing
            pool = Pool(self._parallel)
            ret = pool.map(self.job, list(range(Nmax + 1)))
            pool.close()
            pool.join()

            Cnm = []
            Snm = []

            ret.sort(key=lambda x: x[0])
            for i in range(Nmax + 1):
                Cnm = Cnm + ret[i][1]
                Snm = Snm + ret[i][2]

            return Cnm, Snm

        # for single process
        for n in range(Nmax + 1):
            self.job(n)

        return Cnm, Snm

    def analysis(self, Nmax: int, Inner: list, lat, lon, Pnm, kind=HarAnalysisType.InnerIntegral):
        """
        A simpson-type quadrature method.
        lat,lon, I and Pnm should keep accordance.
        The output is SH coefficients in terms of geo-potential.
        Notice: the ordering of Inner has to be N * M.  (rows sorted by latitude)

        Notice: the grid resolution has indeed determined the max degree (Nmax) of SH expansion, therefore the user
        defined Nmax can not be set without limitation, otherwise the high degree/order of output SH is fake.
        For regular grid that spans for N * N, the Nmax is suggested to be less than N/2.
        For Driscoll-healy grid that spans for N * 2N, the Nmax is suggested to be less than N/2.
        For Gauss-Legendre quadrature grid that spans for (N+1)*(2*N+1), the Nmax is suggested to be less than N.

        :param kind: to let the function knows what type of the input gridded fields is
        :param Nmax: maximal degree of spherical harmonic expansion
        :param Inner: degree-dependent radial integration, len=Nmax or a simple gridded field len=1
                        dimension: [len, N*M]
        :param lat: latitude in degree   [dimension: N]
        :param lon: longitude in degree [dimension: M]
        :param Pnm: Legendre function
        :return: A set of spherical harmonic coefficients up to degree Nmax with one-dimension ordering type
        """
        assert kind in HarAnalysisType
        # -------------------------------------------------------------
        # # extract the one-line lat and lon .
        # lat, lon = GeoMathKit.getCoLatLoninRad(lat, lon)
        #
        # # nlon = 2 * nlat
        # phi = lon[0:nlon]
        #
        # theta = np.zeros(nlat)
        # for i in range(nlat):
        #     theta[i] = lat[i * nlon]
        nlat = len(lat)
        nlon = len(lon)
        '''Co-latitude, longitude in [rad] '''
        theta, phi = GeoMathKit.getCoLatLoninRad(lat, lon)
        # --------------------------------------------------------------

        term1 = np.zeros(Nmax + 1)

        for l in range(0, Nmax + 1):
            term1[l] = 1 + 2 * l

        # kl = self._Ln.getNumber(Nmax, self._LoveMethod)
        # M = self._ellipsoid.Mass
        # R = self._ellipsoid.SemimajorAxis
        # factorSH = 4 * np.pi * R ** 2 / M * (1 + kl) / term1

        factorSH = self._factorHarAnalysis(Nmax=Nmax, kind=kind)

        # ---------------------------------------------------------------

        NMmax = int((Nmax + 1) * (Nmax + 2) / 2)

        factor1 = 2 * np.pi / nlon
        factor2 = 0.25 / nlat
        Cnm, Snm = np.zeros(NMmax), np.zeros(NMmax)

        if len(Inner) == 1:
            Am, Bm = np.zeros((Nmax + 1, nlat)), np.zeros((Nmax + 1, nlat))
            I_new = Inner[0].reshape(-1, nlon)

            for m in range(Nmax + 1):
                Am[m] = factor1 * np.array(I_new * np.mat(np.cos(m * phi)).T).flatten()
                Bm[m] = factor1 * np.array(I_new * np.mat(np.sin(m * phi)).T).flatten()

            thetaS = np.tile(np.sin(theta), (GeoMathKit.getIndex(Nmax, Nmax) + 1, 1))
            Qnm = Pnm * thetaS

            for n in range(Nmax + 1):
                indexM = np.arange(n + 1)
                Cnm[GeoMathKit.getIndex(n, 0):GeoMathKit.getIndex(n, n) + 1] = factorSH[n] * factor2 * \
                                                                               np.sum(Qnm[GeoMathKit.getIndex(n,
                                                                                                              0):GeoMathKit.getIndex(
                                                                                   n, n) + 1] * Am[indexM], 1)
                Snm[GeoMathKit.getIndex(n, 0):GeoMathKit.getIndex(n, n) + 1] = factorSH[n] * factor2 * \
                                                                               np.sum(Qnm[GeoMathKit.getIndex(n,
                                                                                                              0):GeoMathKit.getIndex(
                                                                                   n, n) + 1] * Bm[indexM], 1)
                pass

            return Cnm, Snm

        # -------------------------------------------------------------
        if len(Inner) > 1:
            assert len(Inner) >= (Nmax + 1)

        self.__input = (Nmax, nlat, nlon, Inner, factor1, phi, factorSH, Cnm, Snm, factor2, Pnm, theta)

        if self._parallel > 1:
            # for multiprocessing
            pool = Pool(self._parallel)
            ret = pool.map(self.job, list(range(Nmax + 1)))
            pool.close()
            pool.join()

            Cnm = []
            Snm = []

            ret.sort(key=lambda x: x[0])
            for i in range(Nmax + 1):
                Cnm = Cnm + ret[i][1]
                Snm = Snm + ret[i][2]

            return Cnm, Snm

        # for single process
        for n in range(Nmax + 1):
            self.job(n)

        return Cnm, Snm

    def analysis_new(self, Nmax: int, Gqij, lat, lon, PnmMat, kind=HarAnalysisType.InnerIntegral):
        """
        A simpson-type quadrature method.
        lat,lon, I and Pnm should keep accordance.
        The output is SH coefficients in terms of geo-potential.
        Notice: the ordering of Inner has to be N * M.  (rows sorted by latitude)

        Notice: the grid resolution has indeed determined the max degree (Nmax) of SH expansion, therefore the user
        defined Nmax can not be set without limitation, otherwise the high degree/order of output SH is fake.
        For regular grid that spans for N * N, the Nmax is suggested to be less than N/2.
        For Driscoll-healy grid that spans for N * 2N, the Nmax is suggested to be less than N/2.
        For Gauss-Legendre quadrature grid that spans for (N+1)*(2*N+1), the Nmax is suggested to be less than N.

        :param kind: to let the function knows what type of the input gridded fields is
        :param Nmax: maximal degree of spherical harmonic expansion
        :param Inner: degree-dependent radial integration, len=Nmax or a simple gridded field len=1
                        dimension: [len, N*M]
        :param lat: latitude in degree   [dimension: N]
        :param lon: longitude in degree [dimension: M]
        :param PnmMat: Legendre function, got from GeoMathKit.getPnmMatrix
        :return: A set of spherical harmonic coefficients up to degree Nmax with one-dimension ordering type
        """
        Gqij = np.array(Gqij)
        assert len(np.shape(Gqij)) in [3, 4]

        if len(np.shape(Gqij)) == 3:
            colat_rad, lon_rad = GeoMathKit.getCoLatLoninRad(lat, lon)
            nlat, nlon = len(colat_rad), len(lon_rad)

            factorSH = self._factorHarAnalysis(Nmax=Nmax, kind=kind)

            factor1 = np.ones((nlat, Nmax + 1))
            factor1[:, 0] += 1
            factor1 = 1 / (factor1 * nlon)

            factor2 = np.ones((Nmax + 1, Nmax + 1))
            factor2[:, 0] += 1
            factor2 *= np.pi / (2 * nlat)

            m = np.arange(Nmax + 1)
            g = (m[:, None] @ lon_rad[None, :]).T

            co = np.cos(g)
            so = np.sin(g)

            Am = np.einsum('pij,jm->pim', Gqij, co, optimize='greedy') * factor1
            Bm = np.einsum('pij,jm->pim', Gqij, so, optimize='greedy') * factor1

            Cnms = np.einsum('pim,ilm,i->plm', Am, PnmMat, np.sin(colat_rad), optimize='greedy') * factor2
            Snms = np.einsum('pim,ilm,i->plm', Bm, PnmMat, np.sin(colat_rad), optimize='greedy') * factor2

            Cnms = np.einsum('qlm,l->qlm', Cnms, factorSH, optimize='greedy')
            Snms = np.einsum('qlm,l->qlm', Snms, factorSH, optimize='greedy')

        else:
            Cnms = np.zeros((np.shape(Gqij)[1], Nmax + 1, Nmax + 1))
            Snms = np.zeros((np.shape(Gqij)[1], Nmax + 1, Nmax + 1))

            for l in range(Nmax + 1):
                this_Gqij = Gqij[l]
                Cqlm, Sqlm = self.analysis_new(Nmax, this_Gqij, lat, lon, PnmMat, kind=kind)
                Cnms[:, l, :l + 1] = Cqlm[:, l, :l + 1]
                Snms[:, l, :l + 1] = Sqlm[:, l, :l + 1]

        return Cnms, Snms

    def job(self, n):
        """
        Do not call this function as it is only used for parallel computing.
        :param n:
        :return:
        """
        Nmax, nlat, nlon, I, factor1, phi, factorSH, Cnm, Snm, factor2, Pnm, theta = self.__input

        Am, Bm = np.zeros((Nmax + 1, nlat)), np.zeros((Nmax + 1, nlat))
        I_new = I[n].reshape(-1, nlon)

        c = []
        s = []

        thetaS = np.tile(np.sin(theta), (GeoMathKit.getIndex(Nmax, Nmax) + 1, 1))
        Qnm = Pnm * thetaS

        for m in range(n + 1):
            Am[m] = factor1 * np.array(I_new * np.mat(np.cos(m * phi)).T).flatten()
            Bm[m] = factor1 * np.array(I_new * np.mat(np.sin(m * phi)).T).flatten()

        indexM = np.arange(n + 1)
        Cnm[GeoMathKit.getIndex(n, 0):GeoMathKit.getIndex(n, n) + 1] = factorSH[n] * factor2 * \
                                                                       np.sum(Qnm[GeoMathKit.getIndex(n,
                                                                                                      0):GeoMathKit.getIndex(
                                                                           n, n) + 1] * Am[indexM], 1)
        Snm[GeoMathKit.getIndex(n, 0):GeoMathKit.getIndex(n, n) + 1] = factorSH[n] * factor2 * \
                                                                       np.sum(Qnm[GeoMathKit.getIndex(n,
                                                                                                      0):GeoMathKit.getIndex(
                                                                           n, n) + 1] * Bm[indexM], 1)

        c = list(Cnm[GeoMathKit.getIndex(n, 0):GeoMathKit.getIndex(n, n) + 1])
        s = list(Snm[GeoMathKit.getIndex(n, 0):GeoMathKit.getIndex(n, n) + 1])
        return n, c, s

    @DeprecationWarning
    def job_old(self, n):
        """
        Do not call this function as it is only used for parallel computing.
        :param n:
        :return:
        """
        Nmax, nlat, nlon, I, factor1, phi, factorSH, Cnm, Snm, factor2, Pnm, theta = self.__input

        Am, Bm = np.zeros((Nmax + 1, nlat)), np.zeros((Nmax + 1, nlat))
        I_new = I[n].reshape(-1, nlon)

        c = []
        s = []
        for m in range(n + 1):
            for i in range(nlat):
                Am[m, i] = factor1 * np.sum(np.cos(m * phi) * I_new[i])
                Bm[m, i] = factor1 * np.sum(np.sin(m * phi) * I_new[i])

            Cnm[GeoMathKit.getIndex(n, m)] = factorSH[n] * factor2 * \
                                             np.sum(Pnm[GeoMathKit.getIndex(n, m)] * np.sin(theta) * Am[m])
            Snm[GeoMathKit.getIndex(n, m)] = factorSH[n] * factor2 * \
                                             np.sum(Pnm[GeoMathKit.getIndex(n, m)] * np.sin(theta) * Bm[m])

            c.append(Cnm[GeoMathKit.getIndex(n, m)])
            s.append(Snm[GeoMathKit.getIndex(n, m)])
        return n, c, s

    @DeprecationWarning
    def synthesis_old(self, Cnm, Snm, Nmax, lat, lon, kind=SynthesisType.Pressure):
        """
        A two step synthesis method, see the paper GJI (Nico Sneew)
        :param Cnm: in general, it should be the geo-potential coefficients sorted in one dimension.
        :param Snm:
        :param Nmax: Max degree of harmonic expansion.
        :param lat: geophysical latitude in unit "degree" [dimension : N]
        :param lon: geophysical latitude in unit "degree"[dimension : M]
        :param kind: define the form of synthesis
        :return: grid (nlat*nlon) [dimension N*M]
        """

        # -------------convert the (lat, lon) into mathematical form-----------
        lat, lon = GeoMathKit.getCoLatLoninRad(lat, lon)
        nlat = np.size(lat)
        nlon = np.size(lon)
        # -----------------get Pnm----------------------------------------------
        Pnm = GeoMathKit.getPnm(lat, Nmax)
        # ---------------Compute the Am and Bm---------------------------------
        factor = self.__factorSynthesis(Nmax, kind)
        Am, Bm = np.zeros((Nmax + 1, nlat)), np.zeros((Nmax + 1, nlat))
        for m in range(Nmax + 1):
            for l in range(m, Nmax + 1):
                index = GeoMathKit.getIndex(l, m)
                Am[m] = Am[m] + Pnm[index] * Cnm[index] * factor[l]
                Bm[m] = Bm[m] + Pnm[index] * Snm[index] * factor[l]

        Fout = np.zeros((nlon, nlat))

        for j in range(nlon):
            for m in range(Nmax + 1):
                Fout[j] = Fout[j] + Am[m] * np.cos(m * lon[j]) + Bm[m] * np.sin(m * lon[j])

        F = Fout.T

        return F

    def synthesis(self, Cnm, Snm, Nmax, lat, lon, kind=SynthesisType.Pressure):
        """
        A two-step synthesis method, see the paper GJI (Nico Sneew)
        :param Cnm: in general, it should be the geo-potential coefficients sorted in one dimension.
        :param Snm:
        :param Nmax: Max degree of harmonic expansion.
        :param lat: geophysical latitude in unit "degree" [dimension : N]
        :param lon: geophysical latitude in unit "degree"[dimension : M]
        :param kind: define the form of synthesis
        :return: grid (nlat*nlon) [dimension N*M]
        """
        assert kind in SynthesisType
        # -------------convert the (lat, lon) into mathematical form-----------
        lat, lon = GeoMathKit.getCoLatLoninRad(lat, lon)
        nlat = np.size(lat)
        nlon = np.size(lon)
        # -----------------get Pnm----------------------------------------------
        Pnm = GeoMathKit.getPnm(lat, Nmax)
        # ---------------Compute the Am and Bm---------------------------------
        factor = self.__factorSynthesis(Nmax, kind)
        Am, Bm = np.zeros((Nmax + 1, nlat)), np.zeros((Nmax + 1, nlat))
        for m in range(Nmax + 1):
            for l in range(m, Nmax + 1):
                index = GeoMathKit.getIndex(l, m)
                Am[m] = Am[m] + Pnm[index] * Cnm[index] * factor[l]
                Bm[m] = Bm[m] + Pnm[index] * Snm[index] * factor[l]

        Fout = 0

        for m in range(Nmax + 1):
            co = np.cos(m * lon)
            so = np.sin(m * lon)
            Fout = Fout + np.mat(Am[m]).T * co + np.mat(Bm[m]).T * so

        return np.array(Fout)

    def synthesis_new(self, Cqlm, Sqlm, Nmax, lat, lon, PnmMat, kind=SynthesisType.Pressure):
        """
        A two step synthesis method, see the paper GJI (Nico Sneew)
        :param Cnm: in general, it should be the geo-potential coefficients sorted in one dimension.
        :param Snm:
        :param Nmax: Max degree of harmonic expansion.
        :param lat: geophysical latitude in unit "degree" [dimension : N]
        :param lon: geophysical latitude in unit "degree"[dimension : M]
        :param kind: define the form of synthesis
        :return: grid (nlat*nlon) [dimension N*M]
        """
        colat_rad, lon_rad = GeoMathKit.getCoLatLoninRad(lat, lon)

        assert len(Cqlm) == len(Sqlm)
        Cnms = np.array(Cqlm)
        Snms = np.array(Sqlm)

        factor = self.__factorSynthesis(Nmax, kind)

        Am = np.einsum('ijk,ljk,j->ilk', Cnms, PnmMat, factor)
        Bm = np.einsum('ijk,ljk,j->ilk', Snms, PnmMat, factor)

        m = np.arange(Nmax + 1)
        g = m[:, None] @ lon_rad[None, :]

        co = np.cos(g)
        so = np.sin(g)

        Fout = Am @ co + Bm @ so

        return np.array(Fout)

    def __factorSynthesis(self, Nmax, kind=SynthesisType.synthesis):
        """
        Generate factor for SH coefficients while synthesising kinds of physical fields on the surface
        Notice: this function asks for that the SH coefficients has to be geo-potential ones.
        :param Nmax:
        :param kind:
        :return:
        """

        factor = np.zeros(Nmax + 1)
        termI = np.arange(Nmax + 1)
        term = 2 * termI + 1.

        R = self._ellipsoid.SemimajorAxis
        rho_ave = self._ellipsoid.rho_ave
        rp = R

        if kind == SynthesisType.synthesis:
            factor[:] = 1.
        elif kind == SynthesisType.Geoidheight:
            # factor[:] = R
            factor[:] = self._ellipsoid.GM / self._ellipsoid.SemimajorAxis / self._ellipsoid.je
        elif kind == SynthesisType.EWH:
            kl = self._Ln.getNumber(Nmax, self._LoveMethod)
            factor = (R * rho_ave / 3) * (term / (1 + kl)) / Constants.rho_water
        elif kind == SynthesisType.Pressure:
            kl = self._Ln.getNumber(Nmax, self._LoveMethod)
            factor = (R * rho_ave / 3) * (term / (1 + kl)) * Constants.g_wmo
        elif kind == SynthesisType.Potential:
            factor = self._ellipsoid.GM / rp * ((R / rp) ** termI)
        elif kind == SynthesisType.GravityDisturbance:
            factor = self._ellipsoid.GM / rp / rp * (termI + 1) * ((R / rp) ** termI)
        elif kind == SynthesisType.FreeAirGraviyAnomaly:
            factor = self._ellipsoid.GM / rp / rp * (termI - 1) * ((R / rp) ** termI)

        return factor

    def _factorHarAnalysis(self, Nmax, kind=HarAnalysisType.analysis):
        """
        Generate factor for SH coefficients while analyzing kinds of physical fields.
        Notice: this function asks for that the output SH coefficients should be geo-potential ones.
        :param Nmax:
        :param kind:
        :return:
        """

        factor = np.zeros(Nmax + 1)
        termI = np.arange(Nmax + 1)
        term = 2 * termI + 1.

        R = self._ellipsoid.SemimajorAxis
        rho_ave = self._ellipsoid.rho_ave
        M = self._ellipsoid.Mass

        if kind == HarAnalysisType.analysis:
            factor[:] = 1.
        elif kind == HarAnalysisType.GeoidHeight:
            # factor[:] = 1./R
            factor[:] = 1. / (self._ellipsoid.GM / self._ellipsoid.SemimajorAxis / self._ellipsoid.je)
        elif kind == HarAnalysisType.EWH:
            kl = self._Ln.getNumber(Nmax, self._LoveMethod)
            factor = 1. / ((R * rho_ave / 3) * (term / (1 + kl)) / Constants.rho_water)
            # factor = 4 * np.pi * R ** 2 / M * (1 + kl) / term * Constants.rho_water
        elif kind == HarAnalysisType.Pressure:
            kl = self._Ln.getNumber(Nmax, self._LoveMethod)
            factor = 1. / ((R * rho_ave / 3) * (term / (1 + kl)) * Constants.g_wmo)
        elif kind == HarAnalysisType.InnerIntegral:
            '''inner integral for only pressure type'''
            kl = self._Ln.getNumber(Nmax, self._LoveMethod)
            # factor = 1. / ((R * rho_ave / 3) * (term / (1 + kl)))
            '''equal to 4 * np.pi * R ** 2 / M * (1 + kl) / term1'''
            # factor = 4 * np.pi * R ** 2 / M * (1 + kl) / term
            factor = 1. / ((R * rho_ave / 3) * (term / (1 + kl)))
        elif kind == HarAnalysisType.InnerIntegral_EWH:
            '''inner integral for only pressure type'''
            kl = self._Ln.getNumber(Nmax, self._LoveMethod)
            # factor = 1. / ((R * rho_ave / 3) * (term / (1 + kl)))
            '''equal to 4 * np.pi * R ** 2 / M * (1 + kl) / term1'''
            # factor = 4 * np.pi * R ** 2 / M * (1 + kl) / term * Constants.rho_water
            factor = 1. / ((R * rho_ave / 3) * (term / (1 + kl)) / Constants.rho_water)

        return factor
