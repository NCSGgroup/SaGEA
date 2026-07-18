#!/use/bin/env python
# coding=utf-8
# @Author  : Shuhao Liu
# @Time    : 2026/6/17 13:25 
# @File    : FSC_filtering.py
import pathlib

import cartopy
import h5py
import numpy as np

import sagea
from sagea.utils.TimeTool import TimeTool


def generate_kaula_matrix(l_max=60, power=2.):
    degrees = []
    for l in range(l_max + 1):
        num_coefficients = 2 * l + 1
        degrees.extend([l] * num_coefficients)

    degrees = np.array(degrees)
    N = len(degrees)  # 总系数个数 (例如 l_max=60 时，N = 61^2 = 3721)

    kaula_variances = np.zeros(N)

    for i, l in enumerate(degrees):
        if l == 0:
            kaula_variances[i] = 1

        elif l == 1:
            kaula_variances[i] = 1

        else:
            kaula_variances[i] = 1 / (l ** power)

    K = np.diag(kaula_variances)

    K *= 1e-20

    return K


def demo():
    lmax = 60
    gfc_path_list = []
    for year in range(2002, 2017 + 1):
        gfc_path_list += list(pathlib.Path(
            f"/Volumes/WorkDrive/data/GRACE/L2_SH_products/GSM/CSR/RL06/BA01/{year}/"
        ).glob("GSM-2*_0600"))
    gfc_path_list.sort()

    shc = sagea.SHC.from_gfc(gfc_path_list, lmax=lmax, key='GRCOF2')

    dates_begin, dates_end = TimeTool.match_dates_from_name(gfc_path_list)
    dates_ave = TimeTool.get_average_dates(dates_begin, dates_end)

    shc.de_mean(inplace=True)

    path_vcm = pathlib.Path(
        "/Users/shuhao/PycharmProjects/SaGEA/local/FullConstract/output/VCM_Sig_HIS_full_1995_2020_de_trend_cycle_monthly.npy"
    )
    vcm_sig = np.load(path_vcm)

    path_vcm_err = pathlib.Path(
        "/Volumes/WorkDrive/data/GRACE/L2_error_covariance/ITSG/resorted/n60/Covariance_GSM_ITSG_Grace2018_2002-04.hdf5"
    )
    with h5py.File(path_vcm_err, "r") as f:
        vcm_err = f["value"][:]

    vcm_kaula = generate_kaula_matrix(l_max=lmax, power=4)

    vcm_sig /= np.linalg.trace(vcm_sig)
    vcm_kaula /= np.linalg.trace(vcm_kaula)

    p = 50
    shc.value = shc.value[None, p]

    shc.filter(
        "fsc",
        vcm_err=vcm_err, vcm_sig_list=[vcm_sig, vcm_kaula], init_alphas=[1, 1], from_degree=2, scale=1e1,
        inplace=True
    )

    grids = []

    for shc in (
            shc,
            # shc_geo,
    ):
        shc.convert(from_type="Geopotential", to_type="EWH", inplace=True)
        grid = shc.to_grid(1)
        grid.value *= 100

        grids.append(grid)

    grid = grids[0]

    grid.plot(
        index=[0],
        titles=dates_ave[p:p + 1],
        title='b',
        gridlines=False, vmin=-20, vmax=20,
        projection=cartopy.crs.Robinson()
    )


if __name__ == "__main__":
    demo()
