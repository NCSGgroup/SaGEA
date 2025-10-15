import numpy as np

from pysrc.auxiliary.preference.EnumClasses import PhysicalDimensions
from pysrc.auxiliary.aux_tool.MathTool import MathTool


class Harmonic:
    """
    Harmonic analysis and synthesis: Ordinary 2D integration for computing Spherical Harmonic coefficients
    """

    def __init__(self, lat, lon, lmax: int, option=0, discrete=False):
        """

        :param lat: Co-latitude if option=0, unit[rad]; geophysical latitude if option = others, unit[degree]
        :param lon: If option=0, unit[rad]; else unit[degree]
        :param lmax: int, max degree/order
        :param option:
        :param discrete: bool, if True, the given lat and lon represent each point, and should be of the same length.

        """

        if discrete:
            assert len(lat) == len(lon)

        if option != 0:
            self.lat, self.lon = MathTool.get_colat_lon_rad(lat, lon)
        else:
            self.lat, self.lon = lat, lon

        self.lmax = lmax
        self.__discrete = discrete

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

    def analysis(self, gqij: np.ndarray, special_type: PhysicalDimensions = None, lat_weight="DH"):
        assert len(gqij.shape) in (2, 3)
        assert not self.__discrete, "not support discrete analysis yet"
        assert lat_weight in ("DH", None)

        if not self.__discrete:
            single = (len(gqij.shape) == 2)
            if single:
                gqij = np.array([gqij])

            assert special_type in (
                PhysicalDimensions.HorizontalDisplacementEast, PhysicalDimensions.HorizontalDisplacementNorth, None)

            if special_type in (
                    PhysicalDimensions.HorizontalDisplacementEast, PhysicalDimensions.HorizontalDisplacementNorth):
                assert False, "Horizontal Displacement is not supported yet."

            g = self.g
            co = np.cos(g)  # cos(m phi)
            so = np.sin(g)  # sin(m phi)

            am = np.einsum('pij,mj->pim', gqij, co, optimize='greedy') * self.factor1
            bm = np.einsum('pij,mj->pim', gqij, so, optimize='greedy') * self.factor1

            lat_to_int = self.lat
            if lat_weight == "DH":
                # get latitude weights, J. Driscoll and D. Healy, 1994
                wi_DH = np.ones((self.nlat,))
                for j in range(len(wi_DH)):
                    this_theta = self.lat[j]
                    this_wi = np.sum(np.array(
                        [np.sin((2 * l + 1) * this_theta) / (2 * l + 1) for l in range(int(self.nlat / 2 - 1))]
                    ))
                    wi_DH[j] = this_wi
                wi_DH *= 4 / np.pi
                lat_to_int *= wi_DH

            if special_type is None:
                cqlm = np.einsum('pim,ilm,i->plm', am, self.pilm, np.sin(lat_to_int), optimize='greedy') * self.factor2
                sqlm = np.einsum('pim,ilm,i->plm', bm, self.pilm, np.sin(lat_to_int), optimize='greedy') * self.factor2


            elif special_type == PhysicalDimensions.HorizontalDisplacementNorth:
                """do NOT use this code!"""

                pilm_derivative = MathTool.get_Legendre_derivative(self.lat, self.lmax)

                cqlm = np.einsum('pim,ilm,i->plm', -am, pilm_derivative, np.sin(lat_to_int)) * self.factor2
                sqlm = np.einsum('pim,ilm,i->plm', -bm, pilm_derivative, np.sin(lat_to_int)) * self.factor2

            elif special_type == PhysicalDimensions.HorizontalDisplacementEast:
                """do NOT use this code!"""

                mrange = np.arange(self.lmax + 1)
                pilm_divide_sin_theta = self.pilm / np.sin(lat_to_int)[:, None, None]
                m_pilm_divide_sin_theta = np.einsum("m,ilm->ilm", mrange, pilm_divide_sin_theta)

                am_east = np.einsum('pij,mj->pim', gqij, -so, optimize='greedy') * self.factor1
                bm_east = np.einsum('pij,mj->pim', gqij, co, optimize='greedy') * self.factor1

                cqlm = np.einsum('pim,ilm,i->plm', am_east, m_pilm_divide_sin_theta, np.sin(lat_to_int)) * self.factor2
                sqlm = np.einsum('pim,ilm,i->plm', bm_east, m_pilm_divide_sin_theta, np.sin(lat_to_int)) * self.factor2

            else:
                assert False

            if single:
                assert cqlm.shape[0] == 1 and sqlm.shape[0] == 1
                return cqlm[0], sqlm[0]

            else:
                return cqlm, sqlm

        else:
            assert False

    def synthesis(self, cqlm: np.ndarray, sqlm: np.ndarray, special_type: PhysicalDimensions = None):
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

        if not self.__discrete:
            gqij = am @ co + bm @ so
        else:
            gqij = np.einsum("qil,li->qi", am, co) + np.einsum("qil,li->qi", bm, so)

        if single:
            assert gqij.shape[0] == 1
            return gqij[0]
        else:
            return gqij
