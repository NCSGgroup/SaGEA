from enum import Enum

import numpy as np

from sagea.constant import PhysicalDimension
from sagea.utils import MathTool
from sagea.processing.Quadratures import GLQ, DH2, DH


class GRDType(Enum):
    DH = 1
    DH2 = 2
    GLQ = 3


class Harmonic:
    """
    Spherical Harmonic Calculation under the Definition of Fixed Frame Networks (GLQ, DH, and DH2)
    """

    def __init__(self, lmax, grid_type: GRDType or None = GRDType.GLQ, lat=None, lon=None):
        """
        lat, lon given in [degree]
        """


        self.lmax = lmax

        assert (lat is None and lon is None) or (lat is not None and lon is not None)
        assert (grid_type is None) ^ (lat is None)

        if lat is not None:
            self.__grid_type = None

            self.colat, self.lon = MathTool.get_colat_lon_rad(lat, lon)
            # in unit radians

            '''simplified sin weight'''
            # self.analysis_weight = np.sin(self.colat)

            '''get latitude weights, J. Driscoll and D. Healy, 1994'''
            nlat = len(self.colat)
            wi_DH = np.ones_like(self.colat, )
            for j in range(len(wi_DH)):
                this_theta = self.colat[j]
                this_wi = np.sum(np.array(
                    [np.sin((2 * l + 1) * this_theta) / (2 * l + 1) for l in range(int(nlat / 2 - 1))]
                ))
                wi_DH[j] = this_wi
            wi_DH *= 4 / np.pi
            self.analysis_weight = np.sin(self.colat) * wi_DH

        else:
            self.__grid_type = grid_type
            self.colat, self.lon, self.analysis_weight = self.__get_grid_nodes(lmax, grid_type)  # in unit radians

        self.analysis_weight /= np.sum(self.analysis_weight)  # normalized 1

        self.__prepare()

    @staticmethod
    def __get_grid_nodes(lmax, grid_type: GRDType or None):
        if grid_type == GRDType.GLQ:
            colat, lon, analysis_weight = GLQ.get_nodes(lmax)

        elif grid_type == GRDType.DH2:
            colat, lon, analysis_weight = DH2.get_nodes(lmax)

        elif grid_type == GRDType.DH:
            colat, lon, analysis_weight = DH.get_nodes(lmax)

        else:
            assert False

        return colat, lon, analysis_weight

    def __prepare(self):
        """pre-process some parameters of the 'two-step' method."""
        self.nlat, self.nlon = len(self.colat), len(self.lon)

        self.pilm = MathTool.get_Legendre(self.colat, self.lmax)
        # Associative Legendre polynomials, indexes stand for (co-lat[rad], degree l, order m)
        # 3-d array [theta, l, m] shape: (nlat * (lmax + 1) * (lmax + 1))
        m = np.arange(self.lmax + 1)
        self.g = m[:, None] @ self.lon[None, :]

    def synthesis(self, cqlm: np.ndarray, sqlm: np.ndarray, special_type: PhysicalDimension = None):
        assert cqlm.shape == sqlm.shape
        assert len(cqlm.shape) in (2, 3)

        single = (len(cqlm.shape) == 2)
        if single:
            cqlm = np.array([cqlm])
            sqlm = np.array([sqlm])

        # assert special_type in (
        #     PhysicalDimensions.HorizontalDisplacementEast, PhysicalDimensions.HorizontalDisplacementNorth, None)
        assert special_type is None

        cqlm = np.array(cqlm)
        sqlm = np.array(sqlm)

        if special_type is None:
            am = np.einsum('ijk,ljk->ilk', cqlm, self.pilm)
            bm = np.einsum('ijk,ljk->ilk', sqlm, self.pilm)

        # elif special_type is PhysicalDimensions.HorizontalDisplacementNorth:
        #     pilm_derivative = MathTool.get_Legendre_derivative(self.lat, self.lmax)
        #
        #     am = np.einsum('ijk,ljk->ilk', -cqlm, pilm_derivative)
        #     bm = np.einsum('ijk,ljk->ilk', -sqlm, pilm_derivative)
        #
        # elif special_type is PhysicalDimensions.HorizontalDisplacementEast:
        #     pilm_divide_sin_theta = self.pilm / np.sin(self.lat)[:, None, None]
        #     mrange = np.arange(self.lmax + 1)
        #
        #     am = np.einsum('ijk,k,ljk->ilk', sqlm, mrange, pilm_divide_sin_theta)
        #     bm = np.einsum('ijk,k,ljk->ilk', -cqlm, mrange, pilm_divide_sin_theta)

        else:
            assert False

        co = np.cos(self.g)
        so = np.sin(self.g)

        # gqij = am @ co + bm @ so
        # Note: Unexpected RuntimeWarning (divide/overflow) from NumPy here.
        # Results seem correct, but the warning source is unidentified.

        gqij = np.einsum('ijk,kl->ijl', am, co) + np.einsum('ijk,kl->ijl', bm, so)

        if single:
            assert gqij.shape[0] == 1
            return gqij[0]
        else:
            return gqij

    def analysis(self, gqij: np.ndarray):
        assert len(gqij.shape) in (2, 3)

        single = (len(gqij.shape) == 2)
        if single:
            gqij = np.array([gqij])

        g = self.g
        co = np.cos(g)  # cos(m phi)
        so = np.sin(g)  # sin(m phi)

        factor1 = 1 / self.nlon  # Delta longitude
        am = np.einsum('pij,mj->pim', gqij, co) * factor1
        bm = np.einsum('pij,mj->pim', gqij, so) * factor1

        sin_lat_to_int = self.analysis_weight  # weighted Delta latitude
        cqlm = np.einsum('pim,ilm,i->plm', am, self.pilm, sin_lat_to_int)
        sqlm = np.einsum('pim,ilm,i->plm', bm, self.pilm, sin_lat_to_int)

        if single:
            assert cqlm.shape[0] == 1 and sqlm.shape[0] == 1
            return cqlm[0], sqlm[0]

        else:
            return cqlm, sqlm
