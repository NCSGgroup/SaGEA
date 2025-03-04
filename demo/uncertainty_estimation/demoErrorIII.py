import copy
import itertools
from datetime import date

import matplotlib.pyplot as plt
import numpy as np

from pysrc.auxiliary.aux_tool.FileTool import FileTool
from pysrc.auxiliary.aux_tool.TimeTool import TimeTool
import pysrc.auxiliary.preference.EnumClasses as Enums
from pysrc.auxiliary.load_file.LoadL2LowDeg import load_low_degs
from pysrc.auxiliary.load_file.LoadL2SH import load_SHC


def demo_single_postprocessing(params, low_degs=None, shc_gsm=None, shc_gad=None, shc_gaa=None, shc_gia=None,
                               dates_begin=None, dates_end=None, dates_ave=None, mask=None):
    """"""
    lmax = 60
    grid_space = 1

    '''read config'''
    assert {'replacement', 'decorrelation', 'filtering', 'geometric', 'gia', 'gad', 'gmam', 'leakage'} <= set(
        params.keys())

    replace_deg1 = "deg1" in (params["replacement"])
    replace_c20 = "c20" in (params["replacement"])
    replace_c30 = "c30" in (params["replacement"])

    dec = None if params["decorrelation"] is None else params["decorrelation"][0]
    dec_param = None if params["decorrelation"] is None else params["decorrelation"][1]

    filtering = None if params["filtering"] is None else params["filtering"][0]
    filtering_param = None if params["filtering"] is None else params["filtering"][1]

    geometric = params["geometric"]
    gia = params["gia"]
    gad = params["gad"]
    gmam = params["gmam"]
    leakage = params["leakage"]

    '''begin postprocessing'''
    shc_gsm.replace_low_degs(dates_begin, dates_end, low_deg=low_degs, deg1=replace_deg1, c20=replace_c20,
                             c30=replace_c30)

    if gad == "processed":
        shc_gsm.add(shc_gad, lbegin=1)

    if gmam == "processed":
        shc_gsm.subtract(shc_gaa, lend=0)

    if gia == "processed":
        shc_gsm.subtract(shc_gia)

    if geometric:
        shc_gsm.de_background()
        shc_gsm.geometric(assumption=Enums.GeometricCorrectionAssumption.Ellipsoid)

    shc_gsm.convert_type(from_type=Enums.PhysicalDimensions.Dimensionless, to_type=Enums.PhysicalDimensions.EWH)
    shc_unf = copy.deepcopy(shc_gsm)

    if dec is not None:
        shc_gsm.filter(method=dec, param=dec_param)

    shc_gsm.filter(method=filtering, param=filtering_param)

    if gad == "direct":
        shc_gad.convert_type(from_type=Enums.PhysicalDimensions.Dimensionless, to_type=Enums.PhysicalDimensions.EWH)
        shc_gsm.add(shc_gad, lbegin=1)

    if gmam == "direct":
        shc_gaa.convert_type(from_type=Enums.PhysicalDimensions.Dimensionless, to_type=Enums.PhysicalDimensions.EWH)
        shc_gsm.subtract(shc_gaa, lend=0)

    if gia == "direct":
        shc_gia.convert_type(from_type=Enums.PhysicalDimensions.Dimensionless, to_type=Enums.PhysicalDimensions.EWH)
        shc_gsm.subtract(shc_gia)

    grid = shc_gsm.to_grid(grid_space)

    if leakage == Enums.LeakageMethod.ForwardModeling:
        mask_reverse = 1 - mask
        grid.leakage(method=leakage, basin=mask_reverse, basin_conservation=mask,
                     filter_type=filtering, filter_params=filtering_param, lmax=lmax, )
    else:
        grid.leakage(method=leakage, basin=mask, basin_conservation=mask,
                     filter_type=filtering, filter_params=filtering_param, lmax=lmax,
                     prefilter_type=Enums.SHCFilterType.Gaussian, prefilter_params=(50,),
                     shc_unfiltered=shc_unf)

    gmom = grid.integral(mask)
    gmom -= np.mean(gmom)

    year_fraction = TimeTool.convert_date_format(dates_ave, TimeTool.DateFormat.ClassDate,
                                                 TimeTool.DateFormat.YearFraction)

    return year_fraction, gmom


def demo():
    """this demo shows how to get std of GRACE data by multi-postprocessing"""
    multi_params = {
        "replacement": (
            # ("deg1", "c20"),
            ("deg1", "c20", "c30"),
        ),

        "decorrelation": (
            None,
            (Enums.SHCDecorrelationType.PnMm, (3, 5)),
            # (Enums.SHCDecorrelationType.SlideWindowSwenson2006, (3, 10, 10, 30, 5))
        ),

        "filtering": (
            (Enums.SHCFilterType.Gaussian, (300,)),
            (Enums.SHCFilterType.DDK, (3,)),
            (Enums.SHCFilterType.AnisotropicGaussianHan, (300, 500, 30)),
        ),

        "geometric": (
            # True,
            False,
        ),

        "gad": (
            # "processed",
            "direct",
            False,
        ),

        "gia": (
            "processed",
            "direct",
            # False,
        ),

        "gmam": (
            # "processed",
            # "direct",
            False,
        ),

        "leakage": (
            # Enums.LeakageMethod.ForwardModeling,
            Enums.LeakageMethod.Iterative,
            Enums.LeakageMethod.BufferZone,
        )
    }

    lmax = 60
    grid_space = 1
    begin_date, end_date = date(2005, 1, 1), date(2021, 12, 31)

    '''define filepaths input'''
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

    print("loading files...", end=" ")
    '''load GSM'''
    shc, dates_begin, dates_end = load_SHC(gsm_path, key=gsm_key, lmax=lmax, read_rows=(2, 3, 4, 5),
                                           get_dates=True, begin_date=begin_date, end_date=end_date)
    dates_ave = TimeTool.get_average_dates(dates_begin, dates_end)

    '''load low-degrees'''
    low_degs = load_low_degs(*low_deg_filepaths)

    '''load GAA'''
    shc_gaa, _, _ = load_SHC(gaa_path, key=gaa_key, lmax=lmax, read_rows=(2, 3, 4, 5),
                             get_dates=True, begin_date=begin_date, end_date=end_date)

    '''load GAD'''
    shc_gad, _, _ = load_SHC(gad_path, key=gad_key, lmax=lmax, read_rows=(2, 3, 4, 5),
                             get_dates=True, begin_date=begin_date, end_date=end_date)
    '''load GIA'''
    shc_gia_trend = load_SHC(gia_filepath, key='', lmax=lmax, read_rows=(1, 2, 3, 4))
    shc_gia = shc_gia_trend.expand(dates_ave)

    '''load basin'''
    shc_basin = load_SHC(basin_path, key='', lmax=lmax, read_rows=(1, 2, 3, 4))

    grid_basin = shc_basin.to_grid(grid_space=grid_space)
    grid_basin.limiter(threshold=0.5)
    mask_ocean = grid_basin.value[0]
    print("done!")

    '''get parameters combanations'''
    ranges = []
    keys = list(multi_params.keys())
    for key in keys:
        this_range = list(range(len(multi_params[key])))
        ranges.append(this_range)
    combines = list(itertools.product(*ranges))
    params = []
    for c in range(len(combines)):
        this_params = copy.deepcopy(multi_params)

        for i in range(len(keys)):
            this_params[keys[i]] = multi_params[keys[i]][combines[c][i]]

        params.append(this_params)
    year_fraction = None
    signals = []

    '''begin multi-postprocessing'''
    for i in range(len(params)):
        print("=" * 30)
        this_params = params[i]
        print(f"processing params the {i + 1}th param:\n{this_params} ...\n ({len(params)} in total)")

        x, y = demo_single_postprocessing(
            this_params,
            low_degs=copy.deepcopy(low_degs),
            shc_gsm=copy.deepcopy(shc),
            shc_gad=copy.deepcopy(shc_gad),
            shc_gaa=copy.deepcopy(shc_gaa),
            shc_gia=copy.deepcopy(shc_gia),
            dates_begin=copy.deepcopy(dates_begin),
            dates_end=copy.deepcopy(dates_end),
            dates_ave=copy.deepcopy(dates_ave),
            mask=copy.deepcopy(mask_ocean)
        )

        if year_fraction is None:
            year_fraction = x

        signals.append(y)
    signals = np.array(signals)

    '''statistic'''
    signal_ave = np.mean(signals, axis=0)
    std = np.std(signals, axis=0)

    '''plot'''
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(1, 1, 1)
    ax2 = ax1.twinx()

    ax1.plot(year_fraction, signal_ave * 1000)
    ax2.bar(year_fraction, std * 1000, width=1 / 13, color="red")

    ax1.set_xlabel("Year")
    ax1.set_ylabel("EWHA (mm)")

    ax2.set_ylabel("Std of EWHA (mm)")
    ax2.set_ylim(0, np.max(std * 1000) * 3)

    plt.show()


if __name__ == '__main__':
    demo()
