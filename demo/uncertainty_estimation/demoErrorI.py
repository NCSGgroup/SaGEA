import numpy as np

from pysrc.auxiliary.load_file.LoadL2SH import load_SHC
from pysrc.auxiliary.load_file.LoadCov import load_CovMatrix
import pysrc.auxiliary.preference.EnumClasses as Enums
from pysrc.auxiliary.aux_tool.FileTool import FileTool
from pysrc.auxiliary.scripts.PlotGrids import plot_grids

from pysrc.data_class.DataClass import SHC
from pysrc.uncertainty_estimating.monte_carlo.MonteCarlo import MonteCarlo


def demo1():
    """this demo shows how to get variance of post-processed GRACE var-covariance: a global distribution of one month"""

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

    print("post-processing", end=" ")
    shc_noise.convert_type(from_type=Enums.PhysicalDimensions.Dimensionless, to_type=Enums.PhysicalDimensions.EWH)
    # shc_noise.filter(Enums.SHCDecorrelationType.PnMm, (3, 5))
    shc_noise.filter(Enums.SHCFilterType.DDK, (3,))
    grid_noise = shc_noise.to_grid(grid_space=grid_space)
    print("done!")

    print("get statistic")
    at_index = "diag"
    covar_processed = MonteCarlo.get_covariance(data=grid_noise.value, at_index=at_index)

    plot_grids(
        np.sqrt(covar_processed),
        lat=grid_noise.lat, lon=grid_noise.lon,
    )

    pass


def demo2():
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
    grid_basin = shc_basin.to_grid(grid_space=grid_space)
    grid_basin.limiter(threshold=0.5)
    mask_ocean = grid_basin.value[0]

    shc_noise.convert_type(from_type=Enums.PhysicalDimensions.Dimensionless, to_type=Enums.PhysicalDimensions.EWH)

    filter_method, filter_params = Enums.SHCFilterType.DDK, (3,)
    shc_noise.filter(filter_method, filter_params)
    grid_noise = shc_noise.to_grid(grid_space=grid_space)

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
    demo1()
