import pathlib

import numpy as np

from pysrc.auxiliary.load_file.LoadL2SH import load_SHC
from pysrc.auxiliary.preference.EnumClasses import PhysicalDimensions, SHCFilterType
from pysrc.auxiliary.scripts.PlotGrids import plot_grids
from pysrc.post_processing.harmonic.Harmonic import GRDType


def demo():
    gsm_paths = list(pathlib.Path(
        r"../data/L2_SH_products/GSM/CSR/RL06/BA01/2005"
    ).iterdir())
    lmax = 60
    grid_type = GRDType.GLQ
    grid_space = 1

    shc = load_SHC(*gsm_paths, key="GRCOF2", lmax=lmax)
    shc.convert_type(from_type=PhysicalDimensions.Dimensionless, to_type=PhysicalDimensions.EWH)
    shc.de_background()
    shc.filter(SHCFilterType.Gaussian, (300,))

    # grd = shc.to_GRD(grid_type=grid_type)
    # shc2 = grd.to_SHC()
    # grd2 = shc2.to_GRD(grid_type=grid_type)

    grd = shc.to_GRD()
    shc2 = grd.to_SHC()
    grd2 = shc2.to_GRD()

    # grd = shc.to_GRD(grid_space=grid_space)
    # shc2 = grd.to_SHC(lmax=lmax)
    # grd2 = shc2.to_GRD(grid_space=grid_space)

    print(grd.grid_type)

    max_dif = np.max(np.abs((grd - grd2).value[0]))

    plot_grids(
        (np.array([grd.value[0], grd2.value[0], (grd - grd2).value[0]]) * 100),
        grd.lat, grd.lon,
        vmin=[-20., -20., -max_dif * 100],
        vmax=[20., 20., max_dif * 100],
        title=grd.grid_type
    )


if __name__ == '__main__':
    demo()
