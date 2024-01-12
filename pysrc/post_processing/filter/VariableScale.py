import datetime
from enum import Enum

import numpy as np
from tqdm import trange

from pysrc.auxiliary.core.GRID import GRID
from pysrc.auxiliary.core.SHC import SHC
from pysrc.auxiliary.load_file.LoadL2SH import LoadL2SH
from pysrc.auxiliary.scripts.PlotGrids import plot_grids
from pysrc.auxiliary.tools.FileTool import FileTool
from pysrc.auxiliary.tools.MathTool import MathTool
from pysrc.auxiliary.tools.TimeTool import TimeTool
from pysrc.post_processing.Love_number.LoveNumber import LoveNumber
from pysrc.post_processing.convert_field_physical_quantity.ConvertSHC import ConvertSHC
from pysrc.post_processing.harmonic.Harmonic import Harmonic

from pysrc.auxiliary.preference.Constants import GeoConstants


class VaryRadiusWay(Enum):
    sin = 1
    sin2 = 2


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

    def __init__(self, r_min, r_max=None, sigma=None, harmonic: Harmonic = None,
                 vary_radius_mode: VaryRadiusWay = VaryRadiusWay.sin):

        if r_max is None:
            r_max = r_min

        if sigma is None:
            sigma = np.mat([[1, 0], [0, 1]])

        self.r_min = r_min * 1000
        self.r_max = r_max * 1000
        self.sigma = sigma

        self.vary_way = vary_radius_mode

        self.harmonic = harmonic

        self.radius_e = GeoConstants.radius_earth

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

    def apply_to(self, shc: SHC or GRID, option=0):
        """

        :param shc: SHC if option=0, else GRID.
        :param option:
        :return:
        """
        if option == 0:
            assert self.harmonic is not None

            grid = self.harmonic.synthesis(shc)

        else:
            grid = shc

        Gqij = grid.data
        grid_space = grid.get_grid_space()
        lat, lon = grid.lat, grid.lon

        theta, phi = MathTool.get_colat_lon_rad(lat, lon)

        d_sigma = np.radians(grid_space) ** 2 * np.sin(theta)

        length_of_lat = len(theta)
        length_of_lon = len(phi)
        Wipq = np.zeros((length_of_lat, length_of_lat, length_of_lon))

        Gqij_filtered = np.zeros_like(Gqij)

        for i in range(length_of_lat):
            this_theta = theta[i]
            Wpq = self.get_kernel_at_one_point(grid_space, this_theta, phi[0])

            Wipq[i] = Wpq

        for j in trange(length_of_lon):
            # print('\rfiltering {}/{}'.format(j + 1, length_of_lon), end='\t')

            Gqij_pie = np.zeros_like(Gqij)
            Gqij_pie[:, :, :length_of_lon - j], Gqij_pie[:, :, length_of_lon - j:] = Gqij[:, :, j:], Gqij[:, :, :j]
            Gqij_filtered[:, :, j] = np.einsum('rij,qij->qr', Wipq, Gqij_pie) * d_sigma
        print('done!')

        grid = GRID(Gqij_filtered, lat, lon, option=1)
        if option == 0:
            return self.harmonic.analysis(grid)

        else:
            return grid


def demo():
    """demo for vgc filter"""

    '''load GRACE L2 SH products'''
    load = LoadL2SH()

    load.configuration.set_begin_date(datetime.date(2005, 1, 1))
    load.configuration.set_end_date(datetime.date(2015, 12, 31))

    shc, dates = load.get_shc(with_dates=True)
    shc.de_background()
    ave_dates_GRACE = TimeTool.get_average_dates(*dates)

    convert = ConvertSHC()

    convert.set_Love_number(LoveNumber().get_Love_number())
    shc = convert.apply_to(shc)

    grid_space = 1
    lmax = 60
    lat, lon = MathTool.get_global_lat_lon_range(grid_space)
    har = Harmonic(lat, lon, lmax, option=1)

    rmin, rmax, sigma = 200, 500, np.mat([[1, 0], [0, 0.49]])
    vgc = VariableScale(rmin, rmax, sigma, harmonic=har)

    shc_filtered = vgc.apply_to(shc)

    # grid = har.synthesis(shc)
    grid = har.synthesis(shc_filtered)
    # grid = vgc.apply_to(grid, option=1)

    FileTool.get_project_dir('results/20231128/vgc_spectral/').mkdir(parents=True)
    np.savez(
        FileTool.get_project_dir('results/20231128/vgc_spectral/vgc_spatial_results.npz'),
        grids=grid.data,
        description=f'r_min: {rmin}\n'
                    f'r_max: {rmax}\n'
                    f'r_varying: {vgc.vary_way.name}\n'
                    f'GRACE data: {load.configuration.institute.name} {load.configuration.release.name},'
                    f'begin={load.configuration.beginning_date}, end={load.configuration.ending_date}'
    )

    plot_grids(
        grid.data[0] * 100,
        lat, lon,
        -30, 30,
    )


def demo2():
    """demo: make spectral filtering matrix"""

    '''load GRACE L2 SH products'''
    grid_space = 1
    lmax = 60
    shc = SHC.eye(lmax)

    lat, lon = MathTool.get_global_lat_lon_range(grid_space)
    har = Harmonic(lat, lon, lmax, option=1)

    rmin, rmax, sigma = 200, 500, np.mat([[1, 0], [0, 0.49]])
    vgc = VariableScale(rmin, rmax, sigma, harmonic=har)

    shc_filtered = vgc.apply_to(shc)

    # grid = har.synthesis(shc)
    # grid = har.synthesis(shc_filtered)
    # grid = vgc.apply_to(grid, option=1)

    # FileTool.get_project_dir('results/20231128/vgc_spectral/').mkdir(parents=True)
    np.savez(
        FileTool.get_project_dir('results/20231128/vgc_spectral/vgc_spectral_filtering_matrix_100_300_049.npz'),
        grids=shc_filtered.cs,
        description=f'r_min: {rmin}\n'
                    f'r_max: {rmax}\n'
                    f'r_varying: {vgc.vary_way.name}\n'
    )


def demo3():
    """demo: validate spectral filtering matrix"""

    '''load GRACE L2 SH products'''
    load = LoadL2SH()

    load.configuration.set_begin_date(datetime.date(2005, 1, 1))
    load.configuration.set_end_date(datetime.date(2015, 12, 31))

    shc, dates = load.get_shc(with_dates=True)
    shc.de_background()
    ave_dates_GRACE = TimeTool.get_average_dates(*dates)

    convert = ConvertSHC()

    convert.set_Love_number(LoveNumber().get_Love_number())
    shc = convert.apply_to(shc)

    filtering_matrix = np.load(
        FileTool.get_project_dir('results/20231128/vgc_spectral/vgc_spectral_filtering_matrix.npz')
    )['grids']

    shc.cs = (filtering_matrix @ shc.cs.T).T

    grid_space = 1
    lmax = 60
    lat, lon = MathTool.get_global_lat_lon_range(grid_space)
    har = Harmonic(lat, lon, lmax, option=1)

    grids_spectral_filtering = har.synthesis(shc).data

    grids_spatial_filtering = np.load(
        FileTool.get_project_dir('results/20231128/vgc_spectral/vgc_spatial_results.npz')
    )['grids']

    index = -20
    grids_to_plot = np.array([grids_spatial_filtering[index], grids_spectral_filtering[index],
                              grids_spatial_filtering[index] - grids_spectral_filtering[index]]) * 100
    plot_grids(
        grids_to_plot,
        lat, lon,
        [-50, -50, -5], [50, 50, 5],
        subtitle=['(a) spatial filtered', '(b) spectral filtered', '(a) - (b)']
    )


if __name__ == '__main__':
    demo2()
