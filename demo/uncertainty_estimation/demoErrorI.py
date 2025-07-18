import pathlib
import sys

import cartopy.crs
import matplotlib.pyplot as plt

from pysrc.auxiliary.aux_tool.MathTool import MathTool
from pysrc.auxiliary.load_file.LoadShp import LoadShp
from pysrc.data_class.GRD import GRD

sys.path.append('./')
import numpy as np

from pysrc.auxiliary.load_file.LoadL2SH import load_SHC
from pysrc.auxiliary.load_file.LoadCov import load_CovMatrix
import pysrc.auxiliary.preference.EnumClasses as Enums
from pysrc.auxiliary.aux_tool.FileTool import FileTool
from pysrc.auxiliary.scripts.PlotGrids import plot_grids

from pysrc.data_class.SHC import SHC
from pysrc.uncertainty_estimating.monte_carlo.MonteCarlo import MonteCarlo


def demo_errorI_global_var():
    """this demo shows how to get variance of post-processed GRACE var-covariance: a global distribution of one month"""

    lmax = 60
    grid_space = 1

    print("loading covariance matrix...", end=" ")
    filepath = FileTool.get_project_dir(
        "data/L2_SH_products/VarGSM/ITSG/Grace2018/2009/ITSG-Grace2018_n96_2009-06.snx")
    covmat, date_begin, date_end = load_CovMatrix(filepath, lmax=lmax, get_dates=True)
    print("done!")

    print("generating noises...", end=" ")
    mc = MonteCarlo()
    mc.configuration.sample_num = 1000  # Recommended > 500
    mc.set_input(covmat)
    noise = mc.get_noise()
    shc_noise = SHC(noise)
    print("done!")

    print("post-processing", end=" ")
    shc_noise.convert_type(from_type=Enums.PhysicalDimensions.Dimensionless, to_type=Enums.PhysicalDimensions.EWH)
    # shc_noise.filter(Enums.SHCDecorrelationType.PnMm, (3, 5))
    shc_noise.filter(Enums.SHCFilterType.DDK, (3,))
    grid_noise = shc_noise.to_GRD(grid_space=grid_space)
    print("done!")

    print("get statistic")
    at_index = "diag"
    covar_processed = MonteCarlo.get_covariance(data=grid_noise.value, at_index=at_index)

    plot_grids(
        np.sqrt(covar_processed),
        lat=grid_noise.lat, lon=grid_noise.lon,
    )

    return covar_processed


def demo_errorI_cov_of_points():
    """
    this demo shows how to get var-covariance of post-processed GRACE between given points.
    Based on the provided n one-to-one correspondences lat_each and lon_each, return a covariance matrix with the shape
     of (n, n), whose index order is consistent with the order of the n lat_each and lon_each.
    Note that the lengths of lat_each and lon_each should be consistent.
    """

    lmax = 10
    lat_each = np.array([3.5, 4.5, 3.5, 4.5, 4.5, 5.5, 6.5, 7.5, 5.5, 6.5, 7.5])
    lon_each = np.array([9.5, 9.5, 8.5, 8.5, 7.5, 7.5, 7.5, 7.5, 6.5, 6.5, 6.5])
    assert lat_each.shape == lon_each.shape

    print("loading covariance matrix...", end=" ")
    filepath = FileTool.get_project_dir(
        "data/L2_SH_products/VarGSM/ITSG/Grace2018/2009/ITSG-Grace2018_n96_2009-06.snx")
    covmat, date_begin, date_end = load_CovMatrix(filepath, lmax=lmax, get_dates=True)
    print("done!")

    print("generating noises...", end=" ")
    mc = MonteCarlo()
    mc.configuration.sample_num = 100

    mc.set_input(covmat)
    noise = mc.get_noise()
    shc_noise = SHC(noise)
    print("done!")

    print("post-processing", end=" ")
    shc_noise.convert_type(from_type=Enums.PhysicalDimensions.Dimensionless, to_type=Enums.PhysicalDimensions.EWH)
    shc_noise.filter(Enums.SHCFilterType.DDK, (3,))
    spatial_value_noise = shc_noise.synthesis(lat=lat_each, lon=lon_each, discrete=True)
    print("done!")

    print("get statistic")
    at_index = "full"
    covar_processed = MonteCarlo.get_covariance(data=spatial_value_noise, at_index=at_index)

    """validation"""
    fig = plt.figure(figsize=(6, 4))
    ax1 = fig.add_subplot(1, 2, 1, projection=cartopy.crs.PlateCarree())
    ax2 = fig.add_subplot(1, 2, 2, projection=cartopy.crs.PlateCarree())

    ax1.scatter(lon_each, lat_each, c=np.diag(covar_processed),
                cmap=plt.get_cmap("jet"),
                s=30, marker="s",
                transform=cartopy.crs.PlateCarree()
                )
    ax1.set_title("VAR")

    ax2.scatter(lon_each, lat_each, c=covar_processed[0, :],
                cmap=plt.get_cmap("jet"),
                s=30, marker="s",
                transform=cartopy.crs.PlateCarree()
                )
    ax2.set_title("COV between the first basin")

    for ax in [ax1, ax2]:
        ax.add_feature(cartopy.feature.COASTLINE)
        ax.add_feature(cartopy.feature.LAND)
        ax.add_feature(cartopy.feature.OCEAN)
        ax.set_extent((0, 16, 0, 10))
    fig.show()

    return covar_processed


def demo_errorI_cov_of_basins():
    """
    this demo shows how to get var-covariance of post-processed GRACE between basins.

    Based on n provided base masks, return a covariance matrix with the shape of (n, n),
     whose index order is consistent with the order of the n masks.
    """

    lmax = 60
    grid_space = 1

    shp = LoadShp(path=pathlib.Path("../../data/basin_mask/Shp/bas200k_shp"))
    grid_shp = shp.get_GRD(grid_space=grid_space)
    grid_shp.value = grid_shp.value[:10]
    # As an example, the first ten bases of the sample file are used as the research object here.
    # Users can define their own masks using the following code：
    # lat, lon = MathTool.get_global_lat_lon_range(grid_space)  # lat: -90->90, lon: -180->180 (excluding boundaries)
    # grid_shp = GRD(masks, lat, lon)  # masks: numpy.ndarray in shape (n, 180, 360) for grid_space == 1

    filepath = FileTool.get_project_dir(
        "data/L2_SH_products/VarGSM/ITSG/Grace2018/2009/ITSG-Grace2018_n96_2009-06.snx")
    covmat, date_begin, date_end = load_CovMatrix(filepath, lmax=lmax, get_dates=True)
    print("done!")

    print("generating noises...", end=" ")
    mc = MonteCarlo()

    mc.set_input(covmat)
    noise = mc.get_noise()
    shc_noise = SHC(noise)
    print("done!")

    print("post-processing", end=" ")
    shc_noise.convert_type(from_type=Enums.PhysicalDimensions.Dimensionless, to_type=Enums.PhysicalDimensions.EWH)
    # shc_noise.filter(Enums.SHCDecorrelationType.PnMm, (3, 5))
    shc_noise.filter(Enums.SHCFilterType.DDK, (3,))
    grid_noise = shc_noise.to_GRD(grid_space=grid_space)
    noise_in_basin = grid_noise.regional_extraction(grid_shp)
    print("done!")

    print("get statistic")
    at_index = "full"
    covar_processed = MonteCarlo.get_covariance(data=noise_in_basin.T, at_index=at_index)

    std_map = np.full(grid_shp.value.shape[1:], np.nan)
    cov_map = np.full(grid_shp.value.shape[1:], np.nan)

    for i in range(len(grid_shp)):
        std_map[grid_shp.value[i] == 1] = np.sqrt(covar_processed[i, i]) * 100  # cm
        cov_map[grid_shp.value[i] == 1] = covar_processed[0, i] * 10000  # cm^2, cov with the first (index 0) basin

    """validation"""
    plot_grids(
        std_map, grid_shp.lat, grid_shp.lon, title="STD"
    )
    plot_grids(
        cov_map, grid_shp.lat, grid_shp.lon, title="COV between the first basin"
    )

    return covar_processed


def demo_GMOM():
    """this demo shows how to get variance of post-processed GRACE var-covariance: GMOM of one month"""

    print("loading covariance matrix...", end=" ")
    filepath = FileTool.get_project_dir(
        "data/L2_SH_products/VarGSM/ITSG/Grace2018/2009/ITSG-Grace2018_n96_2009-06.snx")

    lmax = 60
    grid_space = 1
    covmat, date_begin, date_end = load_CovMatrix(filepath, lmax=lmax, get_dates=True)
    print("done!")

    print("generating noises...", end=" ")
    mc = MonteCarlo()

    mc.set_input(covmat)
    noise = mc.get_noise()
    shc_noise = SHC(noise)
    print("done!")

    print("post-processing...", end=" ")
    basin_path = FileTool.get_project_dir("data/basin_mask/SH/Ocean_maskSH.dat")
    shc_basin = load_SHC(basin_path, key='', lmax=lmax, read_rows=(1, 2, 3, 4))
    grid_basin = shc_basin.to_GRD(grid_space=grid_space)
    grid_basin.limiter(threshold=0.5)
    mask_ocean = grid_basin.value[0]

    shc_noise.convert_type(from_type=Enums.PhysicalDimensions.Dimensionless, to_type=Enums.PhysicalDimensions.EWH)

    filter_method, filter_params = Enums.SHCFilterType.DDK, (3,)
    shc_noise.filter(filter_method, filter_params)
    grid_noise = shc_noise.to_GRD(grid_space=grid_space)

    leakage = Enums.LeakageMethod.BufferZone

    if leakage == Enums.LeakageMethod.ForwardModeling:
        mask_land = 1 - mask_ocean
        grid_noise.leakage(method=leakage, basin=mask_land, basin_conservation=mask_ocean,
                           filter_type=filter_method, filter_params=filter_params, lmax=lmax, log=True)

    else:
        grid_noise.leakage(method=leakage, basin=mask_ocean, filter_type=filter_method, filter_params=filter_params,
                           lmax=lmax)

    gmom = grid_noise.integral(mask_ocean)

    print("done!")

    print("getting statistic...", end=" ")
    var_processed = MonteCarlo.get_covariance(data=gmom)
    print("done!")

    print(f"STD of post-processed GRACE covariance is: {np.sqrt(var_processed) * 1000} mm")

    pass


if __name__ == '__main__':
    # demo_errorI_global_var()
    demo_errorI_cov_of_points()
    demo_errorI_cov_of_basins()
