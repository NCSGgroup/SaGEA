import sys
sys.path.append('./')
import copy
from datetime import date

import numpy as np

import pysrc.auxiliary.preference.EnumClasses as Enums

from pysrc.auxiliary.aux_tool.FileTool import FileTool
from pysrc.auxiliary.aux_tool.TimeTool import TimeTool
from pysrc.auxiliary.load_file.LoadL2LowDeg import load_low_degs
from pysrc.auxiliary.load_file.LoadL2SH import load_SHC
from pysrc.auxiliary.load_file.LoadNoah import load_GLDAS_TWS
from pysrc.auxiliary.load_file.LoadShp import LoadShp


def demo1():
    """this demo shows an example of post-processing on global ocean"""

    """To ensure success of verification, please do not change the parameters:"""
    lmax = 60
    grid_space = 1
    begin_date, end_date = date(2009, 1, 1), date(2009, 12, 31)
    gsm_dir, gsm_key = FileTool.get_project_dir("data/L2_SH_products/GSM/CSR/RL06/BA01/"), "GRCOF2"
    gad_dir, gad_key = FileTool.get_project_dir("data/L2_SH_products/GAD/GFZ/RL06/BC01/"), "GRCOF2"
    gaa_dir, gaa_key = FileTool.get_project_dir("data/L2_SH_products/GAA/GFZ/RL06/BC01/"), "GRCOF2"
    low_deg_dir = FileTool.get_project_dir("data/L2_low_degrees/")
    rep_deg1, rep_c20, rep_c30 = True, True, True
    gia_filepath = FileTool.get_GIA_path(gia_type=Enums.GIAModel.Caron2018)
    basin_path_SH = FileTool.get_project_dir("data/basin_mask/SH/Ocean_maskSH.dat")
    geometry_assumption = Enums.GeometricCorrectionAssumption.Ellipsoid
    decorrelation_method, decorrelation_param = Enums.SHCDecorrelationType.PnMm, (3, 10)
    filter_method, filter_params = Enums.SHCFilterType.Gaussian, (500,)
    leakage = Enums.LeakageMethod.Iterative
    """end of input"""

    '''load GSM'''
    shc, dates_begin, dates_end = load_SHC(gsm_dir, key=gsm_key, lmax=lmax, begin_date=begin_date, end_date=end_date,
                                           get_dates=True, )  # load GSM and dates
    dates_ave = TimeTool.get_average_dates(dates_begin, dates_end)  # get average dates

    low_degs = load_low_degs(low_deg_dir, file_id=Enums.L2LowDegreeFileID.TN11, )  # load c20
    low_degs.update(load_low_degs(low_deg_dir, file_id=Enums.L2LowDegreeFileID.TN13,
                                  institute=Enums.L2InstituteType.CSR, release=Enums.L2Release.RL06))  # load degree-1
    low_degs.update(load_low_degs(low_deg_dir, file_id=Enums.L2LowDegreeFileID.TN14))  # load c20 (update) and c30

    shc_gaa = load_SHC(gaa_dir, key=gaa_key, lmax=lmax, begin_date=begin_date, end_date=end_date)  # load GAA
    shc_gad = load_SHC(gad_dir, key=gad_key, lmax=lmax, begin_date=begin_date, end_date=end_date)  # load GAD

    shc_gia_trend = load_SHC(gia_filepath, key='', lmax=lmax)  # load GIA trend
    shc_gia = shc_gia_trend.expand(dates_ave)  # project GIA trend into (monthly) signals along with GRACE times

    shc_basin = load_SHC(basin_path_SH, key='', lmax=lmax)  # load basin mask (in SHC)

    '''replace low degrees'''
    shc.replace_low_degs(dates_begin, dates_end, low_deg=low_degs, deg1=rep_deg1, c20=rep_c20,
                         c30=rep_c30)  # replace SHC low_degrees

    shc.subtract(shc_gia)  # subtracting gia model

    shc.add(shc_gad, lbegin=1)  # recovery of GAD model

    shc.subtract(shc_gaa, lend=0)  # GMAM correction

    shc.de_background()  # de-average if non-input

    shc.geometric(assumption=geometry_assumption, log=True)  # geometric correction, may cost minutes

    shc.convert_type(from_type=Enums.PhysicalDimensions.Dimensionless,
                     to_type=Enums.PhysicalDimensions.EWH)  # convert physical dimension

    shc_unf = copy.deepcopy(shc)
    # shc.filter(method=decorrelation_method, param=decorrelation_param)  # decorrelation filter
    shc.filter(method=filter_method, param=filter_params)  # average filter

    grid = shc.to_GRD(grid_space)  # harmonic synthesis to grid

    # grid.seismic()  # seismic correction

    grid_basin = shc_basin.to_GRD(grid_space=grid_space)
    grid_basin.limiter(threshold=0.5)
    mask_ocean = grid_basin.value[0]  # project basin mask in SHC into grid (1, 0)

    '''leakage reduction'''
    if leakage == Enums.LeakageMethod.ForwardModeling:
        mask_land = 1 - mask_ocean
        grid.leakage(
            method=leakage, basin=mask_land,
            basin_conservation=mask_ocean, filter_type=filter_method, filter_params=filter_params, lmax=lmax,
        )
    else:
        grid.leakage(
            method=leakage, basin=mask_ocean, filter_type=filter_method, filter_params=filter_params, lmax=lmax,
            shc_unfiltered=shc_unf,
        )
    '''end of leakage reduction'''

    gmom = grid.integral(mask_ocean)

    '''validation'''
    validation_path = FileTool.get_project_dir("validation/demo_postprocessing_demo1_ocean.npy")
    gmom_validation = np.load(validation_path)
    if np.max((gmom_validation - gmom) ** 2) < 1e-10:
        print("demo1() for calculating GRACE-based GMOM has been successfully verified!")

    pass


def demo2():
    """this demo shows an example of post-processing on a basin"""

    """To ensure success of verification, please do not change the parameters:"""
    lmax = 60
    grid_space = 1
    begin_date, end_date = date(2009, 1, 1), date(2009, 12, 31)
    gsm_dir, gsm_key = FileTool.get_project_dir("data/L2_SH_products/GSM/CSR/RL06/BA01/"), "GRCOF2"
    low_deg_dir = FileTool.get_project_dir("data/L2_low_degrees/")
    rep_deg1, rep_c20, rep_c30 = True, True, True
    gia_filepath = FileTool.get_GIA_path(gia_type=Enums.GIAModel.Caron2018)
    basin_path_shp, basin_index = FileTool.get_project_dir("data/basin_mask/Shp/bas200k_shp"), 50
    decorrelation_method, decorrelation_param = Enums.SHCDecorrelationType.SlideWindowSwenson2006, (3, 10, 10, 30, 5)
    filter_method, filter_params = Enums.SHCFilterType.DDK, (3,)
    leakage = Enums.LeakageMethod.Addictive
    """end of input"""

    '''load GSM'''
    shc, dates_begin, dates_end = load_SHC(gsm_dir, key=gsm_key, lmax=lmax, begin_date=begin_date, end_date=end_date,
                                           get_dates=True, )  # load GSM and dates
    dates_ave = TimeTool.get_average_dates(dates_begin, dates_end)  # get average dates

    low_degs = load_low_degs(low_deg_dir, file_id=Enums.L2LowDegreeFileID.TN11, )  # load c20
    low_degs.update(load_low_degs(low_deg_dir, file_id=Enums.L2LowDegreeFileID.TN13,
                                  institute=Enums.L2InstituteType.CSR, release=Enums.L2Release.RL06))  # load degree-1
    low_degs.update(load_low_degs(low_deg_dir, file_id=Enums.L2LowDegreeFileID.TN14))  # load c20 (update) and c30

    shc_gia_trend = load_SHC(gia_filepath, key='', lmax=lmax)  # load GIA trend
    shc_gia = shc_gia_trend.expand(dates_ave)  # project GIA trend into (monthly) signals along with GRACE times

    load_shp = LoadShp(basin_path_shp)
    grid_basin = load_shp.get_GRID(grid_space=grid_space)  # load basin mask (in GRID)
    this_basin_mask = grid_basin.value[basin_index]

    grid_gldas, dates_gldas = load_GLDAS_TWS(begin_date, end_date)

    '''replace low degrees'''
    shc.replace_low_degs(dates_begin, dates_end, low_deg=low_degs, deg1=rep_deg1, c20=rep_c20,
                         c30=rep_c30)  # replace SHC low_degrees

    shc.subtract(shc_gia)  # subtracting gia model

    shc.de_background()  # de-average if non-input

    shc.convert_type(from_type=Enums.PhysicalDimensions.Dimensionless,
                     to_type=Enums.PhysicalDimensions.EWH)  # convert physical dimension

    shc_unf = copy.deepcopy(shc)
    shc.filter(method=decorrelation_method, param=decorrelation_param)  # decorrelation filter
    shc.filter(method=filter_method, param=filter_params)  # average filter

    grid = shc.to_GRD(grid_space)  # harmonic synthesis to grid

    grid.leakage(
        method=leakage, basin=this_basin_mask, filter_type=filter_method, filter_params=filter_params, lmax=lmax,
        shc_unfiltered=shc_unf, reference=dict(time=dates_gldas, model=grid_gldas), times=dates_ave
    )  # leakage reduction

    ewh = grid.integral(mask=this_basin_mask)

    '''validation'''
    validation_path = FileTool.get_project_dir("validation/demo_postprocessing_demo2_basin_ewh.npy")
    ewh_validation = np.load(validation_path)
    if np.max((ewh_validation - ewh) ** 2) < 1e-10:
        print("demo2() for calculating GRACE-based regional EWHA has been successfully verified!")

    pass


def demo3():
    """this demo shows an example of post-processing on global distribution"""
    """To ensure a successful verification, please do not change the parameters:"""
    lmax = 60
    grid_space = 1
    begin_date, end_date = date(2005, 1, 1), date(2005, 12, 31)
    gsm_path, gsm_key = FileTool.get_project_dir("data/L2_SH_products/GSM/CSR/RL06/BA01/"), "GRCOF2"
    low_deg_dir = FileTool.get_project_dir("data/L2_low_degrees/")
    gia_filepath = FileTool.get_GIA_path(gia_type=Enums.GIAModel.ICE6GD)
    replace_deg1, replace_c20, replace_c30 = True, True, False
    decorrelation_method, decorrelation_param = Enums.SHCDecorrelationType.SlideWindowStable, (3, 10, 5)
    filter_method, filter_params = Enums.SHCFilterType.Gaussian, (300,)
    """end of input"""

    shc, dates_begin, dates_end = load_SHC(gsm_path, key=gsm_key, lmax=lmax, read_rows=(2, 3, 4, 5),
                                           begin_date=begin_date, end_date=end_date, get_dates=True)
    dates_ave = TimeTool.get_average_dates(dates_begin, dates_end)

    low_degs = load_low_degs(low_deg_dir, file_id=Enums.L2LowDegreeFileID.TN11, )  # load c20
    low_degs.update(load_low_degs(low_deg_dir, file_id=Enums.L2LowDegreeFileID.TN13,
                                  institute=Enums.L2InstituteType.CSR, release=Enums.L2Release.RL06))  # load degree-1
    low_degs.update(load_low_degs(low_deg_dir, file_id=Enums.L2LowDegreeFileID.TN14))  # load c20 (update) and c30

    shc_gia_trend = load_SHC(gia_filepath, key='', lmax=lmax)
    shc_gia = shc_gia_trend.expand(dates_ave)  # project GIA trend into (monthly) signals along with GRACE times

    shc.replace_low_degs(dates_begin, dates_end, low_deg=low_degs, deg1=replace_deg1, c20=replace_c20,
                         c30=replace_c30)  # replace SHC low_degrees

    shc.subtract(shc_gia)  # subtracting gia model

    shc.de_background()  # de-average if non-input

    shc.convert_type(from_type=Enums.PhysicalDimensions.Dimensionless,
                     to_type=Enums.PhysicalDimensions.Pressure)  # convert physical dimension

    shc.filter(method=decorrelation_method, param=decorrelation_param)  # de-correlation filtering
    shc.filter(method=filter_method, param=filter_params)  # average filtering

    grid = shc.to_GRD(grid_space)

    '''validation'''
    validation_path = FileTool.get_project_dir("validation/demo_postprocessing_demo3_global_pressure.npy")
    pressure_validation = np.load(validation_path)
    if np.max((pressure_validation - grid.value) ** 2) < 1e-10:
        print("demo3() for calculating GRACE-based Global distribution of pressure has been successfully verified!")

    pass


if __name__ == '__main__':
    demo1()
