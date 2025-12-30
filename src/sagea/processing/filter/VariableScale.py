from enum import Enum

import numpy as np
from pandas.core.dtypes.inference import is_number
from tqdm import trange

from sagea.utils.MathTool import MathTool
from sagea.constant import GeoConstant


class VaryRadiusWay(Enum):
    """for VGC filter"""
    sin = "sin"
    sin2 = "sin2"


def getPsi(sp, colat, lon):
    """
    sp in unit[degree], co-lat, lon in unit[rad].
    """
    Psi = np.zeros((int(180 / sp), int(360 / sp), 2))

    # colat = np.pi / 2 - lat

    colat_pie = np.pi / 2 - np.radians(np.arange(-90 + sp / 2, 90 + sp / 2, sp))
    delta_lat = colat_pie - colat

    lon_pie = np.radians(np.arange(-180 + sp / 2, 180 + sp / 2, sp))
    delta_lon = lon - lon_pie

    colat_pie, lon_pie = np.meshgrid(colat_pie, lon_pie)

    delta_lat, delta_lon = np.meshgrid(delta_lat, delta_lon)

    # Psi[:, :, 0] = (np.sin(delta_lon / 2) * np.sin((colat_pie + colat) / 2)).T
    Psi[:, :, 0] = (np.sin(delta_lon / 2) * np.sqrt(np.sin(colat_pie) * np.sin(colat))).T
    Psi[:, :, 1] = (np.sin(delta_lat / 2)).T

    return Psi


class VariableScale:
    """
    This class is to smooth grids by applying a spatial convolution with a variable-scale anisotropy Gaussian kernel.
    """

    def __init__(self, r_min, r_max=None, sigma2=None, vary_radius_mode: VaryRadiusWay = None):

        if r_max is None:
            r_max = r_min

        if sigma2 is None:
            sigma2 = np.array([[1, 0], [0, 1]])
        else:
            if is_number(sigma2):
                sigma2 = np.array([[1, 0], [0, sigma2]])
            elif type(sigma2) in (np.ndarray, np.matrix):
                sigma2 = np.array(sigma2)
            else:
                raise ValueError("sigma2 must be as type number, np.ndarray or np.matrix")

        if vary_radius_mode is None:
            vary_radius_mode = VaryRadiusWay.sin

        self.r_min = r_min * 1000
        self.r_max = r_max * 1000
        self.sigma = sigma2

        self.vary_way = vary_radius_mode

        self.radius_e = GeoConstant.radius_earth

    def get_kernel_at_one_point(self, sp, colat, lon):
        """

        :param sp: [degree]
        :param colat: co-lat [rad]
        :param lon: lon [rad]
        :return:
        """

        Psi = getPsi(sp, colat, lon)

        if self.vary_way == VaryRadiusWay.sin2:
            r_lat = (self.r_max - self.r_min) * (np.sin(colat)) ** 2 + self.r_min

        elif self.vary_way == VaryRadiusWay.sin:
            r_lat = (self.r_max - self.r_min) * np.sin(colat) + self.r_min

        else:
            a0 = -0.05771
            b0 = 1.04101

            def fit_func500(x):
                return (a0 * x ** 2 + b0 * x + 500) * 1000

            if colat == np.radians(90):
                r_lat = fit_func500(80)
            else:
                r_lat = fit_func500(np.degrees(np.pi / 2 - colat))

        alpha_0 = r_lat / self.radius_e

        a = np.log(2) / (1 - np.cos(alpha_0))

        sigma = self.sigma

        PsiT_SigmaI_Psi = np.einsum('ijl,lm,ijm->ij', Psi, np.linalg.inv(sigma), Psi)

        weight = a * np.exp(-a * (2 * PsiT_SigmaI_Psi)) / (
                (1 - np.exp(-2 * a)) * 2 * np.pi * np.sqrt(np.linalg.det(sigma)))

        return weight

    def apply_to(self, gqij, sqlm=None):
        """

        :return:
        """
        grid_space = int(180 / np.shape(gqij[0])[0])
        lat, lon = MathTool.get_global_lat_lon_range(grid_space)

        theta, phi = MathTool.get_colat_lon_rad(lat, lon)

        d_sigma = np.radians(grid_space) ** 2 * np.sin(theta)

        length_of_lat = len(theta)
        length_of_lon = len(phi)
        Wipq = np.zeros((length_of_lat, length_of_lat, length_of_lon))

        gqij_filtered = np.zeros_like(gqij)

        for i in range(length_of_lat):
            this_theta = theta[i]

            Wpq = self.get_kernel_at_one_point(grid_space, this_theta, phi[0])

            Wipq[i] = Wpq

        for j in trange(length_of_lon):

            gqij_pie = np.zeros_like(gqij)
            gqij_pie[:, :, :length_of_lon - j], gqij_pie[:, :, length_of_lon - j:] = gqij[:, :, j:], gqij[:, :, :j]
            gqij_filtered[:, :, j] = np.einsum('rij,qij->qr', Wipq, gqij_pie) * d_sigma

        return gqij_filtered
