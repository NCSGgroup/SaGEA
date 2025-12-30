from enum import Enum

import numpy as np

from pysrc.auxiliary.preference.Constants import GeoConstants


class MathTool:
    @staticmethod
    def cs_combine_to_triangle_1d(c: np.ndarray, s: np.ndarray):
        """
        combine the SHC C or S in 2-dimensional matrix to 1-dimension vector,
        or SHC C or S in 3-dimensional matrix to 2-dimension vector.
        Example:

        input
        00
        10 11
        20 21 22
        30 31 32 33

        return cs 1d-array which is formed as
        [c[0,0]; s[1,1], c[1,0], c[1,1]; s[2,2], s[2,1], c[2,0], c[2,1], c[2,2]; s[3,3], s[3,2], s[3,1], c[3,0], ...].
        """
        assert np.shape(c) == np.shape(s)
        lmax = np.shape(c)[-1] - 1

        ones_tril = np.tril(np.ones((lmax + 1, lmax + 1)))
        ones_tri = np.concatenate([ones_tril[:, -1:0:-1], ones_tril], axis=1)
        index_tri = np.where(ones_tri == 1)

        cs_tri = MathTool.cs_combine_to_triangle(c, s)

        if cs_tri.ndim == 2:
            return cs_tri[index_tri]

        elif cs_tri.ndim == 3:
            return np.array([cs_tri[i][index_tri] for i in range(np.shape(cs_tri)[0])])

        else:
            raise Exception

    @staticmethod
    def cs_combine_to_triangle(c: np.ndarray, s: np.ndarray):
        """
        :param c: 2- or 3-d array clm or cqlm
        :param s: 2- or 3-d array slm or sqlm
        return: 2d-array like /s|c\, for example,
        [[0,   ...,   0, c00,   0,  ...,    0],
         [0,   ..., s11, c10, c11,  ...,    0],
         [0,   ..., s21, c20, c21,  ...,    0],
         [...  ...,      ...,       ...,  ...],
         [sii, ..., si1, ci0, ci1,  ...,  cii]]

        or 3d-array with the last two dimensions representing format as above if input are 3-d array.
        """

        assert np.shape(c) == np.shape(s)

        if c.ndim == 2:
            return np.concatenate([s[:, -1:0:-1], c], axis=1)

        elif c.ndim == 3:
            return np.array([np.concatenate([s[i, :, -1:0:-1], c[i]], axis=1) for i in range(np.shape(c)[0])])

        else:
            raise Exception

    @staticmethod
    def cs_decompose_triangle1d_to_cs2d(cs: np.ndarray, fill=0.):
        """
        :param cs: 1d-array sorted as
        [c[0,0]; s[1,1], c[1,0], c[1,1]; s[2,2], s[2,1], c[2,0], c[2,1], c[2,2]; s[3,3], s[3,2], s[3,1], c[3,0], ...].

        :param fill:
        return: 2d-array clm, 2d-array slm
        """
        assert cs.ndim in (1, 2)

        if cs.ndim == 1:
            length_cs1d = len(cs)
            lmax = int(np.sqrt(length_cs1d) - 1)
            shape2d = (lmax + 1, lmax + 1)

            # clm, slm = np.zeros(shape2d), np.zeros(shape2d)
            clm, slm = np.full(shape2d, fill), np.full(shape2d, fill)

            for l in range(lmax + 1):
                for m in range(l + 1):
                    c_index_tri1d = int(l ** 2 + l + m)
                    clm[l, m] = cs[c_index_tri1d]

                    if m > 0:
                        s_index_tri1d = int(l ** 2 + l - m)
                        slm[l, m] = cs[s_index_tri1d]

            return clm, slm

        else:
            cqlm, sqlm = [], []
            for i in range(np.shape(cs)[0]):
                clm, slm = MathTool.cs_decompose_triangle1d_to_cs2d(cs[i])
                cqlm.append(clm)
                sqlm.append(slm)

            return np.array(cqlm), np.array(sqlm)

    @staticmethod
    def get_Legendre(lat, lmax: int, option=0):
        """
        get associated legendre function up to degree/order lmax in Lat. Be careful that this applies the 4-pi normalizaton.
        :param lat: ndarray, co-latitude if option=0, unit[rad]; geophysical latitude if option = others, unit[degree]
        :param lmax: int, max degree
        :param option:
        :return: 3d-ndarray, indexes stand for (co-lat[rad], degree l, order m)
        """

        if option != 0:
            lat = (90. - lat) / 180. * np.pi

        if type(lat) is np.ndarray:
            lsize = np.size(lat)
        else:
            lsize = 1

        pilm = np.zeros((lsize, lmax + 1, lmax + 1))
        pilm[:, 0, 0] = 1
        pilm[:, 1, 1] = np.sqrt(3) * np.sin(lat)

        '''For the diagonal element'''
        for n in range(2, lmax + 1):
            pilm[:, n, n] = np.sqrt((2 * n + 1) / (2 * n)) * np.sin(lat) * pilm[:, n - 1, n - 1]

        for n in range(1, lmax + 1):
            pilm[:, n, n - 1] = np.sqrt(2 * n + 1) * np.cos(lat) * pilm[:, n - 1, n - 1]

        for n in range(2, lmax + 1):
            for m in range(n - 2, -1, -1):
                pilm[:, n, m] = \
                    np.sqrt((2 * n + 1) / ((n - m) * (n + m)) * (2 * n - 1)) \
                    * np.cos(lat) * pilm[:, n - 1, m] \
                    - np.sqrt((2 * n + 1) / ((n - m) * (n + m)) * (n - m - 1) * (n + m - 1) / (2 * n - 3)) \
                    * pilm[:, n - 2, m]

        return pilm

    @staticmethod
    def get_Legendre_1d_index(lat, Nmax: int, option=0):
        """
        get legendre function up to degree/order Nmax in Lat.
        :param lat: Co-latitude if option=0, unit[rad]; geophysical latitude if option = others, unit[degree]
        :param Nmax:
        :param option:
        :return:
        """

        if option != 0:
            lat = (90. - lat) / 180. * np.pi

        NMmax = int((Nmax + 1) * (Nmax + 2) / 2)

        if type(lat) is np.ndarray:
            Nsize = np.size(lat)
        else:
            Nsize = 1

        Pnm = np.zeros((NMmax, Nsize))

        Pnm[MathTool.getIndex(0, 0)] = 1

        Pnm[MathTool.getIndex(1, 1)] = np.sqrt(3) * np.sin(lat)

        '''For the diagonal element'''
        for n in range(2, Nmax + 1):
            Pnm[MathTool.getIndex(n, n)] = np.sqrt((2 * n + 1) / (2 * n)) * np.sin(lat) * Pnm[
                MathTool.getIndex(n - 1, n - 1)]

        for n in range(1, Nmax + 1):
            Pnm[MathTool.getIndex(n, n - 1)] = np.sqrt(2 * n + 1) * np.cos(lat) * Pnm[
                MathTool.getIndex(n - 1, n - 1)]

        for n in range(2, Nmax + 1):
            for m in range(n - 2, -1, -1):
                Pnm[MathTool.getIndex(n, m)] = \
                    np.sqrt((2 * n + 1) / ((n - m) * (n + m)) * (2 * n - 1)) \
                    * np.cos(lat) * Pnm[MathTool.getIndex(n - 1, m)] \
                    - np.sqrt((2 * n + 1) / ((n - m) * (n + m)) * (n - m - 1) * (n + m - 1) / (2 * n - 3)) \
                    * Pnm[MathTool.getIndex(n - 2, m)]

        return Pnm

    @staticmethod
    def get_Legendre_derivative(lat, lmax: int, option=0):
        """
        get derivative of legendre function up to degree/order lmax in Lat.
        :param lat: ndarray, co-latitude if option=0, unit[rad]; geophysical latitude if option = others, unit[degree]
        :param lmax: int, max degree
        :param option:
        :return: 3d-ndarray, indexes stand for (co-lat[rad], degree l, order m)
        """

        # TODO Relationships can be used directly instead of recursively

        def a(nn, mm):
            return np.sqrt((2 * nn + 1) * (2 * nn - 1) / ((nn - mm) * (nn + mm)))

        def b(nn):
            return np.sqrt((2 * nn + 1) / (2 * nn))

        if option != 0:
            lat = (90. - lat) / 180. * np.pi

        pilm = MathTool.get_Legendre(lat, lmax, option=0)

        pilm_d = np.zeros_like(pilm)
        pilm_d[:, 0, 0] = 0
        pilm_d[:, 1, 0] = - np.sqrt(3) * np.sin(lat)
        pilm_d[:, 1, 1] = np.sqrt(3) * np.cos(lat)

        for n in range(2, lmax + 1):
            pilm_d[:, n, n] = b(n) * (
                    np.cos(lat) * pilm[:, n - 1, n - 1] + np.sin(lat) * pilm_d[:, n - 1, n - 1]
            )

            pilm_d[:, n, n - 1] = a(n, n - 1) * (
                    -np.sin(lat) * pilm[:, n - 1, n - 1] + np.cos(lat) * pilm_d[:, n - 1, n - 1]
            )

        for n in range(2, lmax + 1):
            for m in range(0, n - 1):
                pilm_d[:, n, m] = a(n, m) * (
                        -np.sin(lat) * pilm[:, n - 1, m] + np.cos(lat) * pilm_d[:, n - 1, m] - pilm_d[:, n - 2, m] / a(
                    n - 1, m)
                )

        return pilm_d

    @staticmethod
    def get_colat_lon_rad(lat, lon):
        """
        :param lat: geophysical coordinate in degree
        :param lon: geophysical coordinate in degree
        :return: co-latitude and longitude in rad
        """

        theta = (90. - lat) / 180. * np.pi
        phi = lon / 180. * np.pi

        return theta, phi

    @staticmethod
    def get_lat_lon_degree(theta, phi):
        """
        :param theta: co-latitude, geophysical coordinate in rad
        :param phi: longitude, geophysical coordinate in rad
        :return: latitude and longitude in degree
        """

        lat = 90. - theta * 180. / np.pi
        lon = phi * 180. / np.pi

        return lat, lon

    @staticmethod
    def get_global_lat_lon_range(resolution):
        """
        get geophysical latitude and longitude range with a given spatial resolution, i.e., the grid space.
        :param resolution: in unit [degree]
        :return: tuple with two elements of 1d-array, latitude and longitude range in unit [degree]
        """
        lat = np.arange(-90 + resolution / 2, 90 + resolution / 2, resolution)
        lon = np.arange(-180 + resolution / 2, 180 + resolution / 2, resolution)

        return lat, lon

    @staticmethod
    def sort_covariance_matrix(cov_cs, lmax_input: int, lmin_input: int = 0):
        """
        Generally the diagonal index of covariance matrix of spherical harmonic coefficients is sorted by degree like:
        ( var(c00); var(c10), var(c11), var(s11); var(c20), var(c21), var(s21), var(c22), var(s22); ... ),
        or omitted degree-0 and degree-1, like:
        ( var(c20), var(c21), var(s21), var(c22), var(s22); var(s33), var(s32), var(s31), var(c30), ... ),

        Due to the fact that the ranking of spherical harmonic coefficients in this program is based on:
        ( var(c00); var(s11), var(c10), var(c11); var(s22), var(s21), var(c20), var(c21), var(c22); ... ),
        it needs be re-sorted before get further used.

        :param cov_cs: 2d-array given in the above form
        :param lmax_input: max degree/order
        :param lmin_input: the minimum degree/order given in the input

        return: 2d-array that describe the re-sorted covariance matrix of spherical harmonic coefficients
        """
        if lmin_input == 0:
            cov_cs_full = cov_cs
        else:
            cov_cs_full = np.zeros(((lmax_input + 1) ** 2, (lmax_input + 1) ** 2))
            cov_cs_full[lmin_input ** 2:, lmin_input ** 2:] = cov_cs

        lrange = np.arange(1, 121 + 1, 2)

        def get_index2d_from_index1dnew(index):
            l = int(np.sqrt(index))
            m = index - l ** 2 - l
            return l, m

        def get_index1dold(l, m):
            return np.sum(lrange[:l]) - 1 + 2 * np.abs(m) + (1 if m <= 0 else 0)

        identity_matrix = np.eye(np.shape(cov_cs_full)[0])
        elementary_transformation_matrix = np.zeros_like(cov_cs_full)

        '''index 1d (after transform) -> l, m -> index 1d (before transform)'''
        for index in range((lmax_input + 1) ** 2):
            l, m = get_index2d_from_index1dnew(index)
            index_old = get_index1dold(l, m)
            elementary_transformation_matrix[index] = identity_matrix[index_old]

        return elementary_transformation_matrix @ cov_cs_full @ elementary_transformation_matrix.T

    @staticmethod
    def get_design_matrix(function, t):
        m = len(t)
        arg_count = function.__code__.co_argcount - 1
        identity_m = np.eye(m)
        identity_m_extension = [list(identity_m[i]) * arg_count for i in range(m)]
        identity_m_extension = np.array(identity_m_extension).reshape(m * arg_count, m)

        t_extension = np.dot(identity_m_extension, t)
        identity_arg = np.eye(arg_count)
        arg_extension = [np.array(list(identity_arg[i]) * m) for i in range(arg_count)]

        A = function(t_extension, *arg_extension).reshape(m, arg_count)
        return A

    @staticmethod
    def curve_fit(function, t, *y, weight=None):
        """
        if weight is None, OLS;
        else, WLS.
        """

        if len(y) == 1:
            y = y[0].reshape(-1, 1)
        else:
            y = np.vstack(y)

        A = MathTool.get_design_matrix(function, t)

        if weight is not None:
            assert len(t) == len(weight)
            D = np.diag(np.sqrt(weight))
            A = D @ A
            y = D @ y

        AT = A.T
        ATA_I = np.linalg.pinv(np.dot(AT, A))
        A_ginv = np.dot(ATA_I, AT)
        A_ginv_A_ginv_T = np.dot(A_ginv, A_ginv.T)

        results = np.dot(A_ginv, y)

        epsilon = np.dot(A, results) - y

        if np.shape(epsilon)[1] == 1:
            var = np.var(epsilon) * A_ginv_A_ginv_T
        else:
            var = np.array([np.var(epsilon[:, i]) * A_ginv_A_ginv_T for i in range(np.shape(epsilon)[1])])

        return results.T, var

    @staticmethod
    def global_integral(grids, lat=None, lon=None, for_square=False):
        is_single = False
        if len(np.shape(grids)) == 2:
            grids = np.array([grids])

            is_single = True

        radius_e = GeoConstants.radius_earth

        grid_shape = np.shape(grids[0])

        if lat is None:
            lat = np.linspace(-90, 90, grid_shape[0])

        if lon is None:
            lon = np.linspace(-180, 180, grid_shape[1])

        colat_rad, lon_rad = MathTool.get_colat_lon_rad(lat, lon)

        dlat = np.abs(colat_rad[1] - colat_rad[0])
        dlon = np.abs(lon_rad[1] - lon_rad[0])

        domega = np.sin(colat_rad) * dlat * dlon * radius_e ** 2

        if for_square:
            integral = np.einsum('pij,i->p', grids, domega ** 2)
        else:
            integral = np.einsum('pij,i->p', grids, domega)

        if is_single:
            return integral[0]

        else:
            return integral

    @staticmethod
    def get_acreage(basin):

        assert len(basin.shape) in (2, 3)

        if len(basin.shape) == 2:
            acreage = MathTool.global_integral(np.array([basin]))[0]
        elif len(basin.shape) == 3:
            acreage = MathTool.global_integral(basin)
        else:
            assert False

        return acreage

    @staticmethod
    def xyz2grd(xyz):
        """
        xyz to grid
        :param xyz: np.ndarray or list, [ [lon0, lat0, z0], [lon1, lat1, z1], ... ], lons and lats' spacing are equal.
        :return: np.ndarray, grid
        """
        gs = max(np.abs(xyz[0][0] - xyz[1][0]), np.abs(xyz[0][1] - xyz[1][1]))
        lat, lon = np.zeros(int(180 / gs, )), np.zeros((int(360 / gs),))

        grid = np.zeros((int(180 / gs), int(360 / gs)))
        for i in xyz:
            this_lon, this_lat, this_value = i[0], i[1], i[2]

            l, m = MathTool.getGridIndex(this_lat, this_lon, gs)

            lat[l] = this_lat + gs / 2
            lon[m] = this_lon + gs / 2

            grid[l][m] = this_value
        return grid, lat, lon

    @staticmethod
    def getGridIndex(lat, lon, gs):
        return int((lat + 90) / gs), int((lon + 180) / gs)

    @staticmethod
    def getIndex(n: int, m: int):
        """
        index of Cnm at degree-ordered one-dimension array
        :param n: degree
        :param m: order
        :return:
        """
        assert m <= n

        return int(n * (n + 1) / 2 + m)

    @staticmethod
    def get_degree_rms(cqlm, sqlm):
        assert np.shape(cqlm) == np.shape(sqlm)
        if len(np.shape(cqlm)) == 2:
            cqlm = np.array([cqlm])
            sqlm = np.array([sqlm])

        shape = np.shape(cqlm)
        lmax = np.shape(cqlm)[1] - 1

        rms = np.zeros((shape[0], lmax + 1))

        for i in range(lmax + 1):
            rms_this_degree = np.sum(cqlm[:, i, :i + 1] ** 2 + sqlm[:, i, :i + 1] ** 2, axis=1)
            rms_this_degree = np.sqrt(rms_this_degree / ((i + 1) ** 2))
            rms[:, i] = rms_this_degree

        return rms

    @staticmethod
    def get_degree_rss(cqlm, sqlm):
        assert np.shape(cqlm) == np.shape(sqlm)
        if len(np.shape(cqlm)) == 2:
            cqlm = np.array([cqlm])
            sqlm = np.array([sqlm])

        shape = np.shape(cqlm)
        lmax = np.shape(cqlm)[1] - 1

        rss = np.zeros((shape[0], lmax + 1))

        for i in range(lmax + 1):
            rss_this_degree = np.sum(cqlm[:, i, :i + 1] ** 2 + sqlm[:, i, :i + 1] ** 2, axis=1)
            rss_this_degree = np.sqrt(rss_this_degree)
            rss[:, i] = rss_this_degree

        return rss

    @staticmethod
    def get_cumulative_rss(cqlm, sqlm):
        assert np.shape(cqlm) == np.shape(sqlm)
        rss = MathTool.get_degree_rss(cqlm, sqlm)
        crss = np.zeros_like(rss)
        for i in range(rss.shape[1]):
            crss[:, i] = np.sqrt(np.sum(rss[:, :i + 1] ** 2, axis=1))

        return crss
