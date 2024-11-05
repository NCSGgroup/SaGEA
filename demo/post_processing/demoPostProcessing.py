"""
This demo is a validation for GRACE post-processing on global ocean
"""
import copy
from datetime import date

import numpy as np
from matplotlib import pyplot as plt

from pysrc.auxiliary.aux_tool.FileTool import FileTool
from pysrc.auxiliary.aux_tool.MathTool import MathTool
from pysrc.auxiliary.aux_tool.TimeTool import TimeTool
from pysrc.auxiliary.load_file.LoadL2LowDeg import load_low_degs
from pysrc.auxiliary.load_file.LoadL2SH import load_SHC
from pysrc.auxiliary.scripts.PlotGrids import plot_grids


def demo():
    """"""
    lmax = 60
    grid_space = 1
    begin_date, end_date = date(2009, 1, 1), date(2010, 12, 31)

    '''define filepaths input'''

    '''CRA-LICOM'''
    gsm_path = FileTool.get_project_dir("data/L2_SH_products/GSM/CSR/RL06/BA01/")
    gsm_key = "GRCOF2"

    gad_path = FileTool.get_project_dir("data/L2_SH_products/GAD/GFZ/RL06/BC01/")
    gad_key = "GRCOF2"

    gaa_path = FileTool.get_project_dir("data/L2_SH_products/GAA/GFZ/RL06/BC01/")
    gaa_key = "GRCOF2"

    low_deg_filepaths = (
        FileTool.get_project_dir("data/L2_low_degrees/TN-11_C20_SLR_RL06.txt"),
        FileTool.get_project_dir("data/L2_low_degrees/TN-13_GEOC_CSR_RL06.1.txt"),
        FileTool.get_project_dir("data/L2_low_degrees/TN-14_C30_C20_SLR_GSFC.txt")
    )

    '''gia path'''
    gia_filepath = FileTool.get_project_dir("data/GIA/GIA.Caron_et_al_2018.txt")

    basin_path = FileTool.get_project_dir("data/basin_mask/Ocean_maskSH.dat")

    pass

    print("loading files...", end=" ")
    '''load GSM'''
    shc, dates_begin, dates_end = load_SHC(gsm_path, key=gsm_key, lmax=lmax, lmcs_in_queue=(2, 3, 4, 5),
                                           get_dates=True, begin_date=begin_date, end_date=end_date)
    dates_ave = TimeTool.get_average_dates(dates_begin, dates_end)

    '''load low-degrees'''
    low_degs = load_low_degs(*low_deg_filepaths)

    '''load GAA'''
    shc_gaa, _, _ = load_SHC(gaa_path, key=gaa_key, lmax=lmax, lmcs_in_queue=(2, 3, 4, 5),
                             get_dates=True, begin_date=begin_date, end_date=end_date)

    '''load GAD'''
    shc_gad, _, _ = load_SHC(gad_path, key=gad_key, lmax=lmax, lmcs_in_queue=(2, 3, 4, 5),
                             get_dates=True, begin_date=begin_date, end_date=end_date)
    '''load GIA'''
    shc_gia_trend = load_SHC(gia_filepath, key='', lmax=lmax, lmcs_in_queue=(1, 2, 3, 4))
    shc_gia = shc_gia_trend.expand(dates_ave)

    '''load basin'''
    shc_basin = load_SHC(basin_path, key='', lmax=lmax, lmcs_in_queue=(1, 2, 3, 4))
    print("done!")

    print("replacing low-degrees...", end=" ")
    '''replace low degrees'''
    shc.replace_low_degs(dates_begin, dates_end, low_deg=low_degs, deg1=True, c20=True, c30=True)
    print("done!")

    print("GAD recovering...", end=" ")
    shc.add(shc_gad, lbegin=1)
    print("done!")

    print("GMAM correction...", end=" ")
    shc.subtract(shc_gaa, lend=0)
    print("done!")

    print("subtracting gia model...", end=" ")
    shc.subtract(shc_gia)
    print("done!")

    print("de-averaging...", end=" ")
    shc.de_background()
    print("done!")

    # print("geometric correcting... (this step may take times for minutes)", end=" ")
    # shc.geometric(assumption="ellipsoid", log=True)
    # print("done!")

    print("converting SHC type (physical quantity)...", end=" ")
    shc.convert_type(from_type="dimensionless", to_type="ewh")
    print("done!")

    print("filtering", end=" ")
    shc_unf = copy.deepcopy(shc)

    filter_method = "gs"
    filter_params = (500,)
    shc.filter(method=filter_method, param=filter_params)
    print("done!")

    print("harmonic synthesising to grid", end=" ")
    grid = shc.to_grid(grid_space)
    print("done!")

    print("plotting global distribution...", end=" ")
    plot_grids(
        grid.value[:3] * 100, lat=grid.lat, lon=grid.lon, vmin=-20, vmax=20,
    )
    print("done!")

    print("leakagr reduction...", end=" ")
    grid_basin = shc_basin.to_grid(grid_space=grid_space)
    grid_basin.limiter(threshold=0.5)
    mask_ocean = grid_basin.value[0]

    # leakage = "forward_modeling"
    # leakage = "iterative"
    leakage = "buffer"

    if leakage == "forward_modeling":
        mask_land = 1 - mask_ocean
        grid.leakage(method=leakage, basin=mask_land, basin_conservation=mask_ocean,
                     filter_type=filter_method, filter_params=filter_params, lmax=lmax, )
    else:
        grid.leakage(method=leakage, basin=mask_ocean, basin_conservation=mask_ocean,
                     filter_type=filter_method, filter_params=filter_params, lmax=lmax,
                     prefilter_type="gs", prefilter_params=(50,),
                     shc_unfiltered=shc_unf)

    print("done!")

    print("saving files...", end=" ")
    grid.savefile(FileTool.get_project_dir("results/test_ocean.hdf5"), time_dim=dates_ave, rewrite=True)
    print("done!")

    print("extracting basin signal...", end=" ")
    gmom = grid.integral(mask_ocean)
    print("done!")

    print("plotting time series...", end=" ")
    year_fraction = TimeTool.convert_date_format(dates_ave, TimeTool.DateFormat.ClassDate,
                                                 TimeTool.DateFormat.YearFraction)

    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_axes([0.15, 0.15, 0.83, 0.83])
    ax.plot(year_fraction, gmom * 1000)  # mm

    # ax.set_xlim(2009, 2011)
    ax.set_xticks([2009, 2009.5, 2010, 2010.5, 2011], ['2009', '', '2010', '', '2011'])
    ax.set_xlabel("year")
    #
    # ax.set_ylim(-15, 20)
    # ax.set_yticks([-10, 0, 10, 15])
    ax.set_ylabel("EWH (mm)")

    plt.show()
    plt.close()

    print("done!")

    print("OLS fitting...", end=" ")

    def f(x, a, b, c, d):
        return a + b * x + c * np.sin(2 * np.pi * x) + d * np.cos(2 * np.pi * x)

    z = MathTool.curve_fit(f, year_fraction, gmom)
    print("done!")
    print(f"trend: {z[0][0, 1] * 1000} mm/year")
    print(f"annual amplitude: {np.sqrt(z[0][0, 2] ** 2 + z[0][0, 3] ** 2) * 1000} mm")

    return year_fraction, gmom, grid


if __name__ == '__main__':
    demo()
