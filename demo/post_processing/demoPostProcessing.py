import copy
from datetime import date

import numpy as np
from matplotlib import pyplot as plt

from pysrc.auxiliary.aux_tool.FileTool import FileTool
from pysrc.auxiliary.aux_tool.TimeTool import TimeTool
from pysrc.auxiliary.load_file.LoadL2LowDeg import load_low_degs
from pysrc.auxiliary.load_file.LoadL2SH import load_SHC
from pysrc.auxiliary.load_file.LoadNoah import load_GLDAS_TWS
from pysrc.auxiliary.scripts.PlotGrids import plot_grids


def demo1():
    """this demo shows an example of post-processing on global ocean"""

    """To ensure a successful verification, please do not change the parameters:"""
    lmax = 60
    grid_space = 1
    begin_date, end_date = date(2009, 1, 1), date(2009, 12, 31)
    gsm_path, gsm_key = FileTool.get_project_dir("data/L2_SH_products/GSM/CSR/RL06/BA01/"), "GRCOF2"
    gad_path, gad_key = FileTool.get_project_dir("data/L2_SH_products/GAD/GFZ/RL06/BC01/"), "GRCOF2"
    gaa_path, gaa_key = FileTool.get_project_dir("data/L2_SH_products/GAA/GFZ/RL06/BC01/"), "GRCOF2"
    low_deg_filepaths = (FileTool.get_project_dir("data/L2_low_degrees/TN-11_C20_SLR_RL06.txt"),
                         FileTool.get_project_dir("data/L2_low_degrees/TN-13_GEOC_CSR_RL06.1.txt"),
                         FileTool.get_project_dir("data/L2_low_degrees/TN-14_C30_C20_SLR_GSFC.txt"))
    gia_filepath = FileTool.get_project_dir("data/GIA/GIA.Caron_et_al_2018.txt")
    basin_path = FileTool.get_project_dir("data/basin_mask/Ocean_maskSH.dat")
    replace_deg1, replace_c20, replace_c30 = True, True, True
    geometry_assumption = "ellipsoid"  # "sphere", "ellipsoid", "actualearth"
    filter_method, filter_params = "gs", (500,)
    leakage = "iterative"  # "forward_modeling", "iterative", "buffer"
    """end of input"""

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
    shc.replace_low_degs(dates_begin, dates_end, low_deg=low_degs, deg1=replace_deg1, c20=replace_c20, c30=replace_c30)
    print("done!")

    print("subtracting gia model...", end=" ")
    shc.subtract(shc_gia)
    print("done!")

    print("GAD recovering...", end=" ")
    shc.add(shc_gad, lbegin=1)
    print("done!")

    print("GMAM correction...", end=" ")
    shc.subtract(shc_gaa, lend=0)
    print("done!")

    print("de-averaging...", end=" ")
    shc.de_background()
    print("done!")

    print("geometric correcting... (this step may take minutes)", end=" ")
    shc.geometric(assumption=geometry_assumption, log=True)
    print("done!")

    print("converting SHC type (physical quantity)...", end=" ")
    shc.convert_type(from_type="dimensionless", to_type="ewh")
    print("done!")

    print("filtering", end=" ")
    shc_unf = copy.deepcopy(shc)
    shc.filter(method=filter_method, param=filter_params)
    print("done!")

    print("harmonic synthesising to grid", end=" ")
    grid = shc.to_grid(grid_space)
    print("done!")

    print("leakage reduction...", end=" ")
    grid_basin = shc_basin.to_grid(grid_space=grid_space)
    grid_basin.limiter(threshold=0.5)
    mask_ocean = grid_basin.value[0]

    if leakage == "forward_modeling":
        mask_land = 1 - mask_ocean
        grid.leakage(
            method=leakage, basin=mask_land,
            basin_conservation=mask_ocean, filter_type=filter_method, filter_params=filter_params, lmax=lmax,
        )
    else:
        grid.leakage(
            method=leakage, basin=mask_ocean,
            basin_conservation=mask_ocean, filter_type=filter_method, filter_params=filter_params, lmax=lmax,
            prefilter_type="gs", prefilter_params=(50,), shc_unfiltered=shc_unf,
        )

    print("done!")

    print("extracting basin signal...", end=" ")
    gmom = grid.integral(mask_ocean)
    print("done!")

    '''validation'''
    validation_path = FileTool.get_project_dir("validation/demo_postprocessing_demo1_ocean.npy")
    gmom_validation = np.load(validation_path)
    if np.max((gmom_validation - gmom) ** 2) < 0.001:
        print("demo1() for calculating GRACE-based GMOM has been successfully verified!")

    pass


def demo2():
    """this demo shows an example of post-processing on a basin"""
    """To ensure a successful verification, please do not change the parameters:"""
    lmax = 60
    grid_space = 1
    begin_date, end_date = date(2009, 1, 1), date(2009, 12, 31)
    gsm_path, gsm_key = FileTool.get_project_dir("data/L2_SH_products/GSM/CSR/RL06/BA01/"), "GRCOF2"
    low_deg_filepaths = (FileTool.get_project_dir("data/L2_low_degrees/TN-11_C20_SLR_RL06.txt"),
                         FileTool.get_project_dir("data/L2_low_degrees/TN-13_GEOC_CSR_RL06.1.txt"),
                         FileTool.get_project_dir("data/L2_low_degrees/TN-14_C30_C20_SLR_GSFC.txt"))
    gia_filepath = FileTool.get_project_dir("data/GIA/GIA.Caron_et_al_2018.txt")
    basin_path = FileTool.get_project_dir("data/basin_mask/Amazon_maskSH.dat")
    replace_deg1, replace_c20, replace_c30 = True, True, True
    filter_method, filter_params = "ddk", (3,)
    leakage = "addictive"  # "addictive", "multiplicative", "scaling", "data_driven"
    """end of input"""

    '''define filepaths input'''
    print("loading files...", end=" ")
    '''load GSM'''
    shc, dates_begin, dates_end = load_SHC(gsm_path, key=gsm_key, lmax=lmax, lmcs_in_queue=(2, 3, 4, 5),
                                           get_dates=True, begin_date=begin_date, end_date=end_date)
    dates_ave = TimeTool.get_average_dates(dates_begin, dates_end)

    '''load low-degrees'''
    low_degs = load_low_degs(*low_deg_filepaths)

    '''load GLDAS'''  # for leakage reductions
    grid_gldas, dates_gldas = load_GLDAS_TWS(begin_date, end_date, log=True)

    '''load GIA'''
    shc_gia_trend = load_SHC(gia_filepath, key='', lmax=lmax, lmcs_in_queue=(1, 2, 3, 4))
    shc_gia = shc_gia_trend.expand(dates_ave)

    '''load basin'''
    shc_basin = load_SHC(basin_path, key='', lmax=lmax, lmcs_in_queue=(1, 2, 3, 4))
    print("done!")

    print("replacing low-degrees...", end=" ")
    shc.replace_low_degs(dates_begin, dates_end, low_deg=low_degs, deg1=replace_deg1, c20=replace_c20, c30=replace_c30)
    print("done!")

    print("subtracting gia model...", end=" ")
    shc.subtract(shc_gia)
    print("done!")

    print("de-averaging...", end=" ")
    shc.de_background()
    print("done!")

    print("converting SHC type (physical quantity)...", end=" ")
    shc.convert_type(from_type="dimensionless", to_type="ewh")
    print("done!")

    print("filtering", end=" ")
    shc_unf = copy.deepcopy(shc)  # for leakage reduction

    shc.filter(method=filter_method, param=filter_params)
    print("done!")

    print("harmonic synthesising to grid", end=" ")
    grid = shc.to_grid(grid_space)
    print("done!")

    print("leakage reduction...", end=" ")
    grid_basin = shc_basin.to_grid(grid_space=grid_space)
    grid_basin.limiter(threshold=0.5)
    mask = grid_basin.value[0]

    grid.leakage(
        method=leakage, basin=mask, basin_conservation=mask, filter_type=filter_method, filter_params=filter_params,
        lmax=lmax, prefilter_type="gs", prefilter_params=(50,), shc_unfiltered=shc_unf,
        reference=dict(time=dates_gldas, model=grid_gldas.value),
        times=dates_ave, scale_type="trend", log=True, fm_iter_times=30
    )

    print("done!")

    print("extracting basin signal...", end=" ")
    ewh = grid.integral(mask=mask)
    print("done!")

    '''validation'''
    validation_path = FileTool.get_project_dir("validation/demo_postprocessing_demo2_Amazon.npy")
    ewh_validation = np.load(validation_path)
    if np.max((ewh_validation - ewh) ** 2) < 0.001:
        print("demo2() for calculating GRACE-based Amazon EWHA has been successfully verified!")

    pass


def demo3():
    """this demo shows an example of post-processing on global distribution"""
    """To ensure a successful verification, please do not change the parameters:"""
    lmax = 60
    grid_space = 1
    begin_date, end_date = date(2005, 1, 1), date(2005, 12, 31)
    gsm_path, gsm_key = FileTool.get_project_dir("data/L2_SH_products/GSM/CSR/RL06/BA01/"), "GRCOF2"
    low_deg_filepaths = (FileTool.get_project_dir("data/L2_low_degrees/TN-11_C20_SLR_RL06.txt"),
                         FileTool.get_project_dir("data/L2_low_degrees/TN-13_GEOC_CSR_RL06.1.txt"),
                         FileTool.get_project_dir("data/L2_low_degrees/TN-14_C30_C20_SLR_GSFC.txt"))
    gia_filepath = FileTool.get_project_dir("data/GIA/GIA.Caron_et_al_2018.txt")
    replace_deg1, replace_c20, replace_c30 = True, True, True
    filter_method, filter_params = "ddk", (3,)
    """end of input"""

    print("loading files...", end=" ")
    '''load GSM'''
    shc, dates_begin, dates_end = load_SHC(gsm_path, key=gsm_key, lmax=lmax, lmcs_in_queue=(2, 3, 4, 5),
                                           begin_date=begin_date, end_date=end_date, get_dates=True)
    dates_ave = TimeTool.get_average_dates(dates_begin, dates_end)

    '''load low-degrees'''
    low_degs = load_low_degs(*low_deg_filepaths)

    '''load GIA'''
    shc_gia_trend = load_SHC(gia_filepath, key='', lmax=lmax, lmcs_in_queue=(1, 2, 3, 4))
    shc_gia = shc_gia_trend.expand(dates_ave)
    print("done!")

    print("replacing low-degrees...", end=" ")
    '''replace low degrees'''
    shc.replace_low_degs(dates_begin, dates_end, low_deg=low_degs, deg1=replace_deg1, c20=replace_c20, c30=replace_c30)
    print("done!")

    print("subtracting gia model...", end=" ")
    shc.subtract(shc_gia)
    print("done!")

    print("de-averaging...", end=" ")
    shc.de_background()
    print("done!")

    print("filtering", end=" ")
    shc.filter(method=filter_method, param=filter_params)
    print("done!")

    print("converting SHC type (physical quantity)...", end=" ")
    shc.convert_type(from_type="dimensionless", to_type="pressure")
    print("done!")

    print("harmonic synthesising to grid", end=" ")
    grid = shc.to_grid(grid_space)
    print("done!")

    '''validation'''
    validation_path = FileTool.get_project_dir("validation/demo_postprocessing_demo3_global_pressure.npy")
    pressure_validation = np.load(validation_path)
    if np.max((pressure_validation - grid.value) ** 2) < 0.001:
        print("demo3() for calculating GRACE-based Global distribution of pressure has been successfully verified!")

    print("plotting global distribution...", end=" ")  # or save grid and plot/analysis on your own way
    plot_grids(
        grid.value[:3] / 100, lat=grid.lat, lon=grid.lon,
        vmin=-20, vmax=20
    )
    print("done!")

    pass


if __name__ == '__main__':
    demo3()
