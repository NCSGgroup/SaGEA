import copy
import sys

sys.path.append("../../")

import matplotlib.pyplot as plt
import numpy as np
import datetime

from pysrc.auxiliary.aux_tool.MathTool import MathTool
from pysrc.auxiliary.aux_tool.TimeTool import TimeTool
from pysrc.auxiliary.load_file.LoadL2LowDeg import load_low_degs
from pysrc.auxiliary.load_file.LoadL2SH import load_SHC
from pysrc.auxiliary.load_file.LoadNoah import load_GLDAS_TWS
from pysrc.auxiliary.load_file.LoadShp import LoadShp
from pysrc.auxiliary.scripts.PlotGrids import plot_grids
from pysrc.data_class.GRD import GRD

from pysrc.auxiliary.aux_tool.FileTool import FileTool
import pysrc.auxiliary.preference.EnumClasses as Enums


def demo():
    """this demo shows an example of post-processing on global distribution and basin EWH anomalies"""

    '''input'''
    lmax = 60  # max degree/order
    grid_space = 1  # Equal spacing (degree) of L3 gridded output

    begin_date, end_date = datetime.date(2005, 1, 1), datetime.date(2005, 12, 31)
    # The time period of the study

    begin_date_ave, end_date_ave = datetime.date(2004, 1, 1), datetime.date(2010, 12, 31)
    # The time period of the averaging background field

    gsm_path = FileTool.get_project_dir(sub="data/L2_SH_products/GSM/ITSG/Grace2018/n60/")
    # The path (folder) of the gravity field L2 file.
    # Later, the program will load all files in the folder and sub-folder that meet the requirements.
    # Note that function FileTool.get_project_dir(sub=None) can obtain the project path and return a pathlib.Path type,
    # param. sub is optional, indicating the next level path pointed to.

    gsm_key = "gfc"  # The identifier of L2 files, some files may use "GRCOF2" as the identifier.

    low_deg_dir = FileTool.get_project_dir(sub="data/L2_low_degrees/")
    # Path to L2 low degree files (TN-11, TN-13, TN-14, etc.)

    gia_filepath = FileTool.get_project_dir(sub="data/GIA/GIA.ICE-6G_D.txt")
    # Path to GIA files

    replace_deg1, replace_c20, replace_c30 = True, True, False
    # choose whether it is needed replacing deg1, c20 or c30 coefficients.

    decorrelation_method, decorrelation_param = Enums.SHCDecorrelationType.SlideWindowStable, (3, 10, 5)
    # Experimental Decorrelation Filter (EDF) methods and parameters
    # methods contain (with params required, see details in Swenson and Wahr, 2006; Chen et al., 2007, etc.):
    # Enums.SHCDecorrelationType.SlideWindowStable, (p, m, window_length);
    # Enums.SHCDecorrelationType.SlideWindowSwenson2006, (p, m, window_length, A, K);
    # Enums.SHCDecorrelationType.PnMm, (p, m).

    filter_method, filter_params = Enums.SHCFilterType.Gaussian, (300,)
    # Filtering methods and parameters
    # methods contain (with params required,
    # see details in Wahr et al., 1998; Han et al., 2005; Zhang et al., 2009; Kusche et al. 2007, etc.):
    # Enums.SHCDecorrelationType.Gaussian, (r);
    # Enums.SHCDecorrelationType.Fan, (r1, r2);
    # Enums.SHCDecorrelationType.AnisotropicGaussianHan, (r1: int, r2: int, m0: int).
    # Enums.SHCDecorrelationType.DDK, (ddktype: int 1-8).
    """end of input"""

    shc, dates_begin, dates_end = load_SHC(gsm_path, key=gsm_key, lmax=lmax, read_rows=(2, 3, 4, 5),
                                           begin_date=begin_date, end_date=end_date, get_dates=True)
    shc_to_get_ave, dates_begin_to_get_ave, dates_end_to_get_ave = load_SHC(gsm_path, key=gsm_key, lmax=lmax,
                                                                            read_rows=(2, 3, 4, 5),
                                                                            begin_date=begin_date_ave,
                                                                            end_date=end_date_ave, get_dates=True)

    dates_ave = TimeTool.get_average_dates(dates_begin, dates_end)

    low_degs = dict()
    low_degs.update(load_low_degs(low_deg_dir, file_id=Enums.L2LowDegreeFileID.TN14))  # load c20, c30
    low_degs.update(load_low_degs(low_deg_dir, file_id=Enums.L2LowDegreeFileID.TN13,
                                  institute=Enums.L2InstituteType.CSR, release=Enums.L2Release.RL06))  # load degree-1

    shc.replace_low_degs(dates_begin, dates_end, low_deg=low_degs, deg1=replace_deg1, c20=replace_c20,
                         c30=replace_c30)  # replace SHC low_degrees
    shc_to_get_ave.replace_low_degs(dates_begin_to_get_ave, dates_end_to_get_ave, low_deg=low_degs, deg1=replace_deg1,
                                    c20=replace_c20, c30=replace_c30)  # replace SHC low_degrees

    shc_ave = shc_to_get_ave.get_average()
    shc.de_background(shc_ave)

    shc_gia_trend = load_SHC(gia_filepath, key='', lmax=lmax)
    shc_gia = shc_gia_trend.expand(dates_ave)  # project GIA trend into (monthly) signals along with GRACE times
    shc_gia.de_background()  # Subtract the average for GIA SHCs

    shc.subtract(shc_gia)  # subtracting gia model

    shc.convert_type(from_type=Enums.PhysicalDimensions.Dimensionless,
                     to_type=Enums.PhysicalDimensions.EWH)  # convert physical dimension

    shc.geometric(assumption=Enums.GeometricCorrectionAssumption.ActualEarth, log=True)
    # geometric correction, see Yang et al., 2022.
    # log=True to print the progress because it may cost times...

    shc_unf = copy.deepcopy(shc)  # prepare for later use in leakage correction

    shc.filter(method=decorrelation_method, param=decorrelation_param)  # de-correlation filtering
    shc.filter(method=filter_method, param=filter_params)  # average filtering

    grid = shc.to_GRD(grid_space)
    grid.de_aliasing(dates_ave, s2=False, p1=False, s1=False, k2=False, k1=False)
    # Correct the aliasing error by fitting the corresponding period
    # If the testing period is shorter than the correction period, errors may error

    '''
    Here `grid` is the spatial distribution of EWH that you may need.
    `grid` if in type of GRD, with attributes:
        .value (in 3-d numpy.array, unit [m]), 
        .lat (in 1-d numpy.array, unit [deg]),
        .lon (in 1-d numpy.array, unit [deg]).
    Simple scripts `plot_grids()` can help you quickly check the results
     '''
    # plot_grids(
    #     grid.value[:3], grid.lat, grid.lon
    # )

    '''Next, we will further extract basin signals'''

    leakage_method = Enums.LeakageMethod.Iterative
    # leakage methods for your study contain:
    # Enums.LeakageMethod.Multiplicative (Longuevergne et al., 2007).
    # Enums.LeakageMethod.Additive (Klees et al., 2007).
    # Enums.LeakageMethod.Scaling (Landerer et al., 2012).
    # Enums.LeakageMethod.Iterative (Wahr et al., 1998)
    # Enums.LeakageMethod.DataDriven (Vishwakarma et al., 2017).
    # for the former three, it requires the dependent model (GLDAS, for example),
    # and note that the length of model dataset should be equal to that of GSM dataset.
    # Different methods may focus on different types of leakage (leak-in or leak-out),
    # please choose according to the actual situation.

    grid_gldas, dates_gldas = load_GLDAS_TWS(begin_date, end_date)

    basin_path_shp = FileTool.get_project_dir("data/basin_mask/Shp/bas200k_shp")

    load_shp = LoadShp(basin_path_shp)
    grid_basin = load_shp.get_GRD(grid_space=grid_space)  # load basin mask (in GRID)
    basin_mask = grid_basin.value[0]
    # for shp (polygon) mask file

    # basin_mask = np.load(FileTool.get_project_dir("data/basin_mask/grids/Eyre_maskGrid.dat(180,360).npy"))
    # grid_basin = GRD(basin_mask, lat=grid.lat, lon=grid.lon)
    # for gridded mask file

    grid.leakage(
        method=leakage_method, basin=basin_mask, filter_type=filter_method, filter_params=filter_params, lmax=lmax,
        shc_unfiltered=shc_unf, reference=dict(time=dates_gldas, model=grid_gldas), times=dates_ave
    )  # leakage correction

    ewh = grid.regional_extraction(grid_basin, average=True)
    # average=True means that the summed result (TWS) is divided by the area to obtain EWH

    '''
    Here `ewh` is the basin EWH that you may need.
    `ewh` if in 2-d numpy.array, with index 0 indicating the basin ID, and index 1 indicating the values.
    Simple scripts of pyplot can help you quickly check the results.
     '''
    # year_frac = TimeTool.convert_date_format(dates_ave, output_type=TimeTool.DateFormat.YearFraction)
    # plt.plot(year_frac, ewh[0])
    # plt.show()


if __name__ == '__main__':
    demo()
